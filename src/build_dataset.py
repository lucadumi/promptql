"""Build the NL->SQL *training* set by template-based synthetic generation.

Why generate instead of hand-write?
  Fine-tuning a model to memorise 20 answers proves nothing. We want a *training*
  set that teaches the same SQL *skills* the eval set tests (projection, COUNT/
  AVG/SUM/MAX/MIN, WHERE on numbers/strings/dates, ORDER BY, LIMIT, DISTINCT,
  GROUP BY, HAVING) but with *different questions and values*, so improvement on
  the held-out eval reflects genuine generalisation.

Why several phrasings per pattern?
  The first version of this generator gave every SQL pattern exactly ONE question
  wording. The resulting fine-tune hit 100% on the in-template eval but fell to
  75% once the same intents were reworded, i.e. it had latched onto the template
  strings. So every pattern below now ships a list of PHRASINGS (different
  register, synonyms, and sentence shape: "List the names of ...", "Which
  employees ...", "Name the staff whose ...") and the generator expands the
  pattern over all of them. Same SQL targets, many ways to ask.

Why magnitude wordings and a contrastive SUM?
  Paraphrase augmentation left exactly two failures, and they were the same
  mistake: "what's our total headcount?" and "how big is the Sales team?" both
  produced SUM(salary) instead of COUNT(*). The model had learned that words of
  magnitude ("total", "how big") mean "add up the money". The fix is not to teach
  those two sentences -- it is to teach the *distinction*, so the count patterns
  now ship size/headcount wordings ("how large is the {dept} team?", "what is the
  headcount for {dept}?") and a contrastive per-department SUM pattern ("what is
  the {dept} department's total payroll?") pins "total" to the noun being
  totalled (people vs money) rather than to the word itself.

Why multi-table JOINs?
  No eval set covered a JOIN, so nothing in the training data taught one. The
  join families below cover the two shapes this project's schemas use: joining on
  a text key (employees.department = departments.name) and joining a table to
  itself (employees.manager_id = employees.id). They are graded by
  data/eval/text2sql_eval_join.jsonl, with the cross-schema counterpart
  (data/eval/text2sql_eval_join_bookstore.jsonl) joining on an integer foreign key
  the training data never shows -- deliberately, so the join result is a
  generalisation measurement rather than a memorisation one.

The four rules this script enforces (the honest-evaluation contract):
  1. SAME schema + SAME canonical SQL style as `data/eval` (see src/data_utils.py
     and src/metrics.py) so train/eval prompt+target formats are identical.
  2. NO LEAKAGE: drop any generated pair whose (normalised) question OR
     (normalised) SQL collides with an eval example. The SQL check reuses
     `src.metrics.normalize_sql` -- the *exact* function used for scoring -- so a
     training target can never equal a graded eval answer string. The blocklist is
     built from EVERY file in data/eval (in-template, paraphrase and cross-schema),
     not just the in-template one, because adding paraphrased training questions
     would otherwise risk colliding with the paraphrase eval.
  3. NOT EVEN CLOSE: report the highest word-overlap (Jaccard) between any training
     question and any eval question, so "no leakage" is a measured claim rather
     than an exact-string technicality. The closest pairs only ever differ in a
     literal value (a different department, threshold or limit), and rule 2
     guarantees their SQL targets are disjoint from every graded answer, so they
     teach the model to read the parameter rather than memorise the exam.
  4. BALANCED: cap how many examples any one pattern contributes. Parameter pools
     differ wildly in size (12 departments x 4 phrasings vs a single GROUP BY
     target), so naive expansion made some patterns 14x more frequent than others
     and the model started answering rarer patterns with the shape of a common
     one. Capping keeps the pattern mix roughly even.
  5. Reproducible: everything is seeded, so re-running yields the same split.

Note on what the leakage filter implies: a pattern whose SQL is *itself* an eval
gold (e.g. `SELECT name FROM departments`) is removed from training entirely, so
the model has to reach those answers by generalising from neighbouring patterns.
That is the point of the exercise, and it is why the pattern balance matters.

Output (JSONL, identical record schema to the eval file -> reuses load_jsonl):
    data/train/text2sql_train.jsonl
    data/train/text2sql_val.jsonl     # stratified 10% held out to watch eval loss

Run it (from the repo root, inside the venv):
    python -m src.build_dataset                 # write train/val + print a report
    python -m src.build_dataset --val-frac 0.15 # bigger validation split
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Make `src` importable whether run as `-m src.build_dataset` or as a file path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_jsonl  # noqa: E402
from src.metrics import normalize_sql  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = REPO_ROOT / "data" / "eval"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "train"

# A kept question may look like an eval question (same template, different
# literal) but must never BE one. Identical word sets are treated as a leak.
IDENTICAL_OVERLAP = 1.0

# Ceiling on examples contributed by any single pattern. Parameter pools differ
# wildly in size, so without this the most-parameterised patterns dominate and
# the model answers rarer patterns with a frequent pattern's shape.
MAX_PER_CATEGORY = 24


def default_eval_files() -> List[Path]:
    """Every eval set on disk, so new ones are de-leaked automatically."""
    return sorted(DEFAULT_EVAL_DIR.glob("*.jsonl"))

# A generated example carries a `category` tag (the template it came from). The
# tag is used only for the stratified split + the report; it is NOT written to
# the JSONL, which keeps the on-disk schema identical to the eval file.
Example = Tuple[str, str, str]  # (category, question, sql)

# ---------------------------------------------------------------------------
# Parameter pools. We deliberately avoid the exact literals used in the eval set
# (salary 100000, budget 500000, date 2020-01-01, top-5, having-10) to spend
# fewer candidates on pairs the leakage filter would only throw away. The filter
# is still the real guarantee -- these lists are just an efficiency nicety.
# ---------------------------------------------------------------------------
DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance", "Human Resources",
    "Operations", "Support", "Research", "Legal", "Product", "Design", "IT",
]
LOCATIONS = [
    "New York", "San Francisco", "London", "Berlin", "Tokyo", "Austin",
    "Seattle", "Boston", "Chicago", "Remote", "Paris", "Toronto",
]
SALARY_THRESHOLDS = [40000, 50000, 60000, 70000, 80000, 90000,
                     110000, 120000, 130000, 150000, 175000, 200000]
BUDGET_THRESHOLDS = [100000, 200000, 250000, 300000, 400000,
                     600000, 750000, 800000, 1000000, 1500000]
HIRE_DATES = ["2017-01-01", "2018-01-01", "2018-06-01", "2019-01-01",
              "2019-06-01", "2021-01-01", "2021-06-01", "2022-01-01",
              "2022-06-01", "2023-01-01"]
TOP_N = [3, 5, 10, 15]
# Thresholds a HAVING question has to copy out of the question. Deliberately a
# mix of one- and two-digit values: when the join patterns below drew only from
# the single-digit end, the fine-tune started truncating "more than 10" to
# "> 1", i.e. it had learned the digit count rather than the number. The join
# families therefore take the two-digit tail (HAVING_N[2:]) to keep the mix even.
HAVING_N = [3, 5, 15, 20, 25]

AGGS = {"AVG": "average", "MAX": "highest", "MIN": "lowest", "SUM": "total"}

# Pools for the construct families added after the first blind measurement (see
# the "reaching past the old syllabus" section below). Years are strings because
# SQLite's strftime returns TEXT, and comparing it to a bare integer silently
# never matches -- the kind of bug that makes a query run and return nothing.
HIRE_YEARS = ["2017", "2018", "2019", "2020", "2022", "2023"]
# Initials / fragments for LIKE. Kept to letters and word-parts that read like a
# real question ("names starting with A", "departments with 'ing' in the name").
NAME_INITIALS = ["A", "B", "C", "D", "E", "J", "M", "P", "R", "S", "T", "W"]
NAME_ENDINGS = ["a", "e", "n", "r", "y"]
DEPT_FRAGMENTS = ["ing", "Sale", "Mark", "Fin", "Res", "Op"]
# Inclusive ranges for BETWEEN. Written as (low, high) pairs rather than drawn
# from the threshold pools so the two bounds are always a sensible band apart.
SALARY_BANDS = [(50000, 90000), (60000, 120000), (70000, 110000),
                (80000, 140000), (90000, 160000), (45000, 75000)]
BUDGET_BANDS = [(200000, 700000), (250000, 850000), (150000, 550000),
                (350000, 950000)]
DATE_BANDS = [("2018-01-01", "2019-12-31"), ("2019-01-01", "2021-12-31"),
              ("2017-06-01", "2020-06-01"), ("2021-01-01", "2023-12-31")]

# ---------------------------------------------------------------------------
# Phrasings. Each list holds interchangeable ways of asking for the SAME SQL.
# The generator expands every pattern over every phrasing, so each SQL shape is
# seen in several registers (formal, imperative, conversational) instead of one
# canonical sentence. This is the fix for a fine-tune that scored 100% on
# in-template questions and 75% on reworded ones.
# ---------------------------------------------------------------------------
P_PROJECT_EMP = [
    "List the {phrase} of all employees.",
    "What are the {phrase} of all employees?",
    "Show me the {phrase} for every employee.",
    "I need the {phrase} of everyone on staff.",
    "Return the {phrase} for all staff members.",
]
# The employees.department projection gets its own wordings. Phrasing it as
# "the departments of all employees" taught the model that the bare word
# "departments" implies a `department` column, and it then answered "List all
# department names" with `SELECT department FROM departments`, a column that does
# not exist. Every phrasing here ties the column to a single employee instead.
P_PROJECT_EMP_DEPT = [
    "Show the department of each employee.",
    "Which department is each employee in?",
    "Return each employee's department.",
    "For every employee, show which department they work in.",
]
P_PROJECT_DEPT = [
    "List the {phrase} of all departments.",
    "What are the {phrase} of all departments?",
    "Show me the {phrase} for every department.",
    "Give me the {phrase} of each department.",
    "Return the {phrase} for all departments.",
]
P_SELECT_STAR_DEPT = [
    "Show all employees in the {dept} department.",
    "Retrieve every column for employees in {dept}.",
    "Give me the complete rows for the {dept} team.",
    "I want all the details of the {dept} staff.",
]
P_COUNT_ALL = [
    "How many {table} are there?",
    "Count the total number of {table}.",
    "What is the total number of {table} on record?",
    "Give me a count of all {table}.",
    "How many rows are in the {table} table?",
    "Tell me the size of the {table} table.",
    # Magnitude wordings: "how large / how big / overall size" is a COUNT, not a
    # SUM. See the module docstring -- this is the fix for "total headcount".
    "How large is the {table} table?",
    "How big is our list of {table}?",
    "What is the overall size of the {table} table?",
]
P_COUNT_DEPT = [
    "How many employees are in the {dept} department?",
    "Count the employees who work in the {dept} department.",
    "How many people work in {dept}?",
    "What is the staff count for {dept}?",
    "Tell me the number of employees assigned to {dept}.",
    # Same magnitude lesson, scoped to one department: asking how *big* a team is
    # counts people. The contrastive P_SUM_IN_DEPT below adds up their salaries.
    "How large is the {dept} team?",
    "What is the headcount for {dept}?",
    "What is the size of the {dept} team?",
    "How big is our {dept} group?",
    "Give me the total headcount in {dept}.",
]
# Contrastive partner for the magnitude wordings above: same "total" word, but a
# money noun, so the model has to read *what* is being totalled instead of firing
# SUM(salary) at every sentence containing "total".
P_SUM_IN_DEPT = [
    "What is the total salary paid in the {dept} department?",
    "Add up the salaries of the {dept} team.",
    "What is the {dept} department's total payroll?",
]
P_AGG_SALARY = [
    "What is the {word} salary of all employees?",
    "Find the {word} salary among all employees.",
    "Across the whole company, what is the {word} salary?",
    "Compute the {word} salary over all employees.",
]
P_AGG_BUDGET = [
    "What is the {word} budget across all departments?",
    "Find the {word} budget of all departments.",
    "Compute the {word} budget over every department.",
]
P_WHERE_SALARY_GT = [
    "List the names of employees who earn more than {n}.",
    "Which employees are paid above {n}? Give their names.",
    "Name the staff whose salary exceeds {n}.",
]
P_WHERE_SALARY_LT = [
    "List the names of employees who earn less than {n}.",
    "Which employees are paid below {n}? Give their names.",
    "Name the staff whose salary is under {n}.",
]
P_WHERE_DATE_GT = [
    "List the names of employees hired after {d}.",
    "Which employees started after {d}? Names only.",
    "Name everyone whose hire date is later than {d}.",
]
P_WHERE_DATE_LT = [
    "List the names of employees hired before {d}.",
    "Which employees started before {d}? Names only.",
    "Name everyone whose hire date is earlier than {d}.",
]
P_ORDER_BY = [
    "Show the names of employees ordered by {col} in {word} order.",
    "Sort the employees by {col} {word} and list their names.",
    "List employee names sorted by {col}, {word}.",
]
P_DISTINCT = [
    "How many distinct {plural} are there in the {table} table?",
    "Count the unique {plural} in the {table} table.",
    "How many different {plural} appear in {table}?",
    "What is the number of distinct {plural} in {table}?",
    "Tell me how many separate {plural} exist in {table}.",
]
P_WHERE_BUDGET_GT = [
    "Show all departments with a budget over {n}.",
    "Which departments have a budget above {n}? Show every column.",
    "Give me the full rows for departments funded over {n}.",
]
P_WHERE_BUDGET_LT = [
    "Show all departments with a budget under {n}.",
    "Which departments have a budget below {n}? Show every column.",
    "Give me the full rows for departments funded under {n}.",
]
P_FILTER_ORDER = [
    "List the names of employees in the {dept} department ordered by name.",
    "Sort the {dept} staff by name and show their names.",
    "Give me the {dept} employees' names in alphabetical order.",
]
P_AGG_IN_DEPT_AVG = [
    "What is the average salary in the {dept} department?",
    "What does the typical {dept} employee earn?",
    "Compute the mean salary for the {dept} team.",
]
P_AGG_IN_DEPT_MAX = [
    "What is the highest salary in the {dept} department?",
    "Find the top salary within the {dept} team.",
    "What is the maximum pay in {dept}?",
]
P_COUNT_LOCATION = [
    "How many departments are located in {loc}?",
    "Count the departments based in {loc}.",
    "How many departments operate out of {loc}?",
    "Tell me the number of departments situated in {loc}.",
]
P_EXTREME_MAX = [
    "Find the name of the employee with the highest salary.",
    "Who earns the most? Give the name.",
    "Name the top earner in the company.",
    "Which employee sits at the top of the pay scale?",
]
P_EXTREME_MIN = [
    "Find the name of the employee with the lowest salary.",
    "Who earns the least? Give the name.",
    "Name the employee at the bottom of the pay scale.",
    "Which employee has the smallest salary?",
]
# More "extreme row" targets, on other columns and the other table. Asking for a
# *row* ("which employee / which department") has to stay clearly distinct from
# asking for a *value* ("what is the highest salary"), which is an ORDER BY ...
# LIMIT 1 versus a bare MAX(). With only the salary pair, and one of those two
# targets removed by the leakage filter, the pattern was the thinnest in the set
# and the model started answering "who is our lowest-paid employee" with
# MIN(salary).
P_EXTREME_RECENT = [
    "Find the name of the most recently hired employee.",
    "Who joined the company most recently? Give the name.",
    "Name the newest hire.",
    "Which employee has the latest hire date?",
]
P_EXTREME_EARLIEST = [
    "Find the name of the longest-serving employee.",
    "Who was hired first? Give the name.",
    "Name the employee with the earliest hire date.",
    "Which employee has been here the longest?",
]
P_EXTREME_DEPT_MAX = [
    "Find the name of the department with the largest budget.",
    "Which department is the best funded? Give its name.",
    "Name the department at the top of the budget list.",
    "Which department has the highest budget?",
]
P_EXTREME_DEPT_MIN = [
    "Find the name of the department with the smallest budget.",
    "Which department is the least funded? Give its name.",
    "Name the department at the bottom of the budget list.",
    "Which department has the lowest budget?",
]
P_TOP_N_DESC = [
    "List the names of the top {n} highest paid employees.",
    "Give me the names of the {n} employees with the largest salaries.",
    "Which {n} employees earn the most? Names only.",
]
P_TOP_N_ASC = [
    "List the names of the {n} lowest paid employees.",
    "Give me the names of the {n} employees with the smallest salaries.",
    "Which {n} employees earn the least? Names only.",
]
P_GROUP_COUNT = [
    "Show each department and the number of employees in it.",
    "For every department, give the employee count.",
    "Group the employees by department and count them.",
]
# The leakage filter removes the un-filtered `department, COUNT(*)` target
# entirely, because that exact SQL is an eval gold. That left the model with no
# example of "grouping column + COUNT(*)" *on the employees table* at all, and a
# lexically adjacent projection ("show the department of each employee" ->
# SELECT department FROM employees) won the tie -- the last in-taxonomy failure
# on the dev sets. These two families restore the shape without ever emitting the
# graded answer: one keeps the grouping column and adds a WHERE, the other keeps
# the table and changes the grouping column. Both are ordinary questions in their
# own right, and neither normalises to any eval gold.
P_GROUP_COUNT_FILTERED = [
    "For each department, how many employees earn more than {n}?",
    "Show each department and the number of employees in it paid above {n}.",
    "Group the employees earning over {n} by department and count them.",
    "Per department, give the count of staff on more than {n}.",
]
P_GROUP_COUNT_MANAGER = [
    "Show each manager id and the number of employees reporting to them.",
    "For every manager, give the number of direct reports.",
    "Group the employees by manager and count them.",
    "Per manager id, how many employees report in?",
]
P_GROUP_AGG = [
    "For each department, show the {phrase}.",
    "Group by department and report the {phrase}.",
    "Per department, what is the {phrase}?",
]
# The same GROUP BY lesson on the other table, so "group and report the
# aggregate" is not tied to one table. Without it the pattern was thin enough
# that the model answered "break down the number of employees by department"
# with a bare GROUP BY and no COUNT(*) column.
P_GROUP_LOC_COUNT = [
    "Show each location and the number of departments in it.",
    "For every location, give the department count.",
    "Group the departments by location and count them.",
]
P_GROUP_LOC_AGG = [
    "For each location, show the {phrase} of the departments there.",
    "Group the departments by location and report the {phrase}.",
    "Per location, what is the {phrase} of the departments?",
]
P_HAVING_GT = [
    "List departments that have more than {n} employees.",
    "Which departments employ more than {n} people?",
    "Show the departments with more than {n} staff members.",
    "Name the departments with over {n} employees.",
]
P_HAVING_LT = [
    "List departments that have fewer than {n} employees.",
    "Which departments employ fewer than {n} people?",
    "Show the departments with under {n} staff members.",
    "Name the departments with less than {n} employees.",
]

# ---------------------------------------------------------------------------
# Reaching past the old syllabus.
#
# The first blind measurement (data/eval/text2sql_eval_blind_v1_retired.jsonl)
# put a number on something the development sets structurally could not see: of
# its six failures, four needed a construct that appears in NO training template
# -- a year extracted from a date, a bare SELECT DISTINCT, an IS NULL test, and
# de-duplication of a join result. The model was not getting those wrong, it had
# never been shown them.
#
# The families below close that gap at the level of the *construct*, never at the
# level of the questions that exposed it. That distinction is the whole protocol:
# v1 has been retired into the development loop precisely so that teaching from
# it is legitimate, and the replacement blind set (v2) was authored independently
# and before any of this existed. As always, the leakage filter still deletes any
# generated pair whose SQL equals a graded answer, so the specific queries v1
# asks for remain things the model has to reach by generalising.
#
# One SQLite-specific lesson is deliberate: `YEAR(hire_date)` is the natural
# thing to write and does not exist in SQLite, so the date family teaches
# `strftime('%Y', ...)` compared against a *quoted* year.
# ---------------------------------------------------------------------------
# Contrastive partner of P_DISTINCT: same word "distinct", but "how many" asks
# for COUNT(DISTINCT col) and "which ones" asks for SELECT DISTINCT col. Teaching
# only the counting half is what made the model drop DISTINCT when asked to list.
P_SELECT_DISTINCT = [
    "List the distinct {plural} in the {table} table.",
    "What are the unique {plural} in {table}?",
    "Show me the different {plural} that appear in {table}, without repeats.",
    "Give me each of the {plural} in {table} exactly once.",
    "Which {plural} appear in {table}? List each one only once.",
]
P_SELECT_DISTINCT_MANAGER = [
    "List the distinct manager ids in the employees table.",
    "Which manager ids appear in employees? Each one only once.",
    "Give me the unique manager ids on record, without repeats.",
]
# Filtered halves of the same contrast, sharing one parameter pool so the only
# thing separating them is "how many" vs "which ones". The unfiltered
# `SELECT DISTINCT location FROM departments` target is removed by the leakage
# filter (it is a graded answer), which would otherwise leave this family thin
# enough to be answered with a neighbouring pattern's shape.
P_SELECT_DISTINCT_WHERE = [
    "Which departments have someone earning more than {n}? List each once.",
    "List the distinct departments containing an employee paid above {n}.",
    "Give me the unique departments where anyone earns over {n}.",
]
P_COUNT_DISTINCT_WHERE = [
    "How many distinct departments have someone earning more than {n}?",
    "Count the different departments containing an employee paid above {n}.",
    "How many separate departments have anyone earning over {n}?",
]
P_NULL_MISSING = [
    "List the names of employees who have no manager.",
    "Which employees have no manager assigned? Names only.",
    "Name the staff whose manager field is empty.",
    "Who does not report to anyone? Give their names.",
]
P_NULL_PRESENT = [
    "List the names of employees who do have a manager.",
    "Which employees have a manager on record? Names only.",
    "Name the staff whose manager field is filled in.",
]
P_NULL_COUNT_MISSING = [
    "How many employees have no manager?",
    "Count the employees with no manager assigned.",
    "How many staff members are missing a manager?",
]
P_NULL_COUNT_PRESENT = [
    "How many employees have a manager on record?",
    "Count the employees whose manager field is filled in.",
    "How many staff members do report to someone?",
]
P_NULL_DEPT = [
    "List the names of departments with no location recorded.",
    "Which departments are missing a location? Names only.",
    "Name the departments whose location field is empty.",
]
P_NULL_DEPT_PRESENT = [
    "Show all departments that do have a location recorded.",
    "Which departments have a location on file? Show every column.",
    "Give me the full rows for departments whose location is not empty.",
]
P_DATE_YEAR = [
    "List the names of employees hired in {y}.",
    "Which employees joined in {y}? Names only.",
    "Name everyone who started in the year {y}.",
    "Who was hired during {y}? Give their names.",
]
P_DATE_YEAR_COUNT = [
    "How many employees were hired in {y}?",
    "Count the staff who joined in {y}.",
    "How many people started during the year {y}?",
]
P_LIKE_PREFIX = [
    "List the names of employees whose name starts with {x}.",
    "Which employees have a name beginning with {x}? Names only.",
    "Name the staff whose name begins with the letter {x}.",
]
P_LIKE_SUFFIX = [
    "List the names of employees whose name ends with {x}.",
    "Which employees have a name ending in {x}? Names only.",
    "Name the staff whose name finishes with the letter {x}.",
]
P_LIKE_CONTAINS = [
    "List the names of departments whose name contains {x}.",
    "Which departments have {x} somewhere in their name? Names only.",
    "Name the departments with {x} inside the name.",
]
P_BETWEEN_SALARY = [
    "List the names of employees earning between {a} and {b}.",
    "Which employees are paid in the {a} to {b} range? Names only.",
    "Name the staff whose salary falls between {a} and {b}.",
]
P_BETWEEN_BUDGET = [
    "Show all departments with a budget between {a} and {b}.",
    "Which departments are funded in the {a} to {b} range? Show every column.",
    "Give me the full rows for departments whose budget sits between {a} and {b}.",
]
P_BETWEEN_DATE = [
    "List the names of employees hired between {a} and {b}.",
    "Which employees joined between {a} and {b}? Names only.",
    "Name everyone whose hire date falls between {a} and {b}.",
]
P_NOT_EQUAL_DEPT = [
    "List the names of employees who are not in the {dept} department.",
    "Which employees work outside {dept}? Names only.",
    "Name the staff who do not belong to {dept}.",
]
P_NOT_EQUAL_LOC = [
    "Show all departments that are not located in {loc}.",
    "Which departments sit outside {loc}? Show every column.",
    "Give me the full rows for departments not based in {loc}.",
]

# ---------------------------------------------------------------------------
# Multi-table JOINs. The employees schema is denormalised, so the two tables are
# related by a TEXT key (employees.department = departments.name) rather than an
# integer foreign key; employees also references itself through manager_id. Both
# shapes are taught here. Every join target is written with fully-qualified
# column names so the model learns which table each column comes from.
# ---------------------------------------------------------------------------
JOIN_DEPT = "FROM employees JOIN departments ON employees.department = departments.name"
JOIN_SELF = "FROM employees e JOIN employees m ON e.manager_id = m.id"

P_JOIN_PROJECT = [
    "List each employee's {ecol} together with the {dcol} of their department.",
    "Show each employee's {ecol} and their department's {dcol}.",
    "For every employee, give their {ecol} and the {dcol} of the department they work in.",
    "Pair each employee's {ecol} with the {dcol} of their department.",
]
P_JOIN_WHERE_LOC = [
    "List the names of employees who work in departments located in {loc}.",
    "Which employees work in {loc}-based departments? Names only.",
    "Name the staff whose department sits in {loc}.",
]
P_JOIN_WHERE_BUDGET_GT = [
    "List the names of employees whose department has a budget over {n}.",
    "Which employees belong to departments funded above {n}? Names only.",
    "Name the staff working in departments with a budget greater than {n}.",
]
P_JOIN_WHERE_BUDGET_LT = [
    "List the names of employees whose department has a budget under {n}.",
    "Which employees belong to departments funded below {n}? Names only.",
    "Name the staff working in departments with a budget smaller than {n}.",
]
P_JOIN_COUNT_LOC = [
    "How many employees work in departments located in {loc}?",
    "Count the employees whose department is based in {loc}.",
    "How many people work out of the {loc} departments?",
]
P_JOIN_AGG_LOC = [
    "What is the {word} salary among employees in departments located in {loc}?",
    "Find the {word} salary across the {loc} departments.",
    "Among staff in {loc}-based departments, what is the {word} salary?",
]
P_JOIN_GROUP_AGG = [
    "For each department location, show the location and the {phrase} of the employees working there.",
    "Group employees by their department's location and report the {phrase}.",
    "Per department location, what is the {phrase}?",
]
P_JOIN_GROUP_COUNT = [
    "For each department location, show the location and how many employees work there.",
    "Group the employees by their department's location and count them.",
    "Per department location, give the employee count.",
]
P_JOIN_HAVING_GT = [
    "List the department locations that have more than {n} employees.",
    "Which locations employ more than {n} people across their departments?",
    "Show the department locations with over {n} staff members.",
]
P_JOIN_HAVING_LT = [
    "List the department locations that have fewer than {n} employees.",
    "Which locations employ fewer than {n} people across their departments?",
    "Show the department locations with under {n} staff members.",
]
P_JOIN_ORDER = [
    "List the names of employees in {loc} departments, ordered by salary in {word} order.",
    "Sort the staff in {loc}-based departments by salary {word} and list their names.",
    "Show employee names for the {loc} departments, sorted by salary {word}.",
]
P_JOIN_TOP1 = [
    "Find the name of the highest paid employee working in a department located in {loc}.",
    "Who earns the most among the staff in {loc}-based departments? Give the name.",
    "Name the top earner across the {loc} departments.",
]
P_SELF_JOIN_PROJECT = [
    "List each employee's name along with their manager's {mcol}.",
    "Show every employee's name and the {mcol} of their manager.",
    "For each employee who has a manager, give the employee's name and the manager's {mcol}.",
]
P_SELF_JOIN_COUNT = [
    "How many employees have a manager?",
    "Count the employees who report to someone.",
    "How many staff members have a manager assigned?",
]
P_SELF_JOIN_WHERE = [
    "List the names of employees whose manager works in the {dept} department.",
    "Which employees report to a manager in {dept}? Names only.",
    "Name the staff whose manager belongs to {dept}.",
]
# DISTINCT over a join: a manager appears once per report, so listing "the people
# who manage someone" without DISTINCT repeats them. This is the join-flavoured
# half of the SELECT DISTINCT lesson above.
P_SELF_JOIN_DISTINCT = [
    "List the names of employees who manage someone, each name only once.",
    "Which staff members have at least one direct report? List each name once.",
    "Give me the distinct names of the people who manage somebody.",
]


def generate_candidates() -> List[Example]:
    """Expand every template over its phrasings + parameter pool."""
    out: List[Example] = []

    def add(cat: str, phrasings: List[str], sql: str, **fmt: object) -> None:
        """Emit one (category, question, sql) row per phrasing of this pattern."""
        for template in phrasings:
            out.append((cat, template.format(**fmt), sql))

    # -- projection: single column from a table -----------------------------
    for phrase, col in [("salaries", "salary"), ("hire dates", "hire_date"),
                        ("names", "name")]:
        add("project_emp", P_PROJECT_EMP, f"SELECT {col} FROM employees", phrase=phrase)
    add("project_emp", P_PROJECT_EMP_DEPT, "SELECT department FROM employees")
    for phrase, col in [("names", "name"), ("budgets", "budget"),
                        ("locations", "location")]:
        add("project_dept", P_PROJECT_DEPT, f"SELECT {col} FROM departments", phrase=phrase)

    # -- SELECT * filtered by department ------------------------------------
    for dept in DEPARTMENTS:
        add("select_star_dept", P_SELECT_STAR_DEPT,
            f"SELECT * FROM employees WHERE department = '{dept}'", dept=dept)

    # -- COUNT(*) whole table -----------------------------------------------
    for table in ["employees", "departments"]:
        add("count_all", P_COUNT_ALL, f"SELECT COUNT(*) FROM {table}", table=table)

    # -- COUNT(*) filtered by department ------------------------------------
    for dept in DEPARTMENTS:
        add("count_dept", P_COUNT_DEPT,
            f"SELECT COUNT(*) FROM employees WHERE department = '{dept}'", dept=dept)

    # -- aggregates over salary / budget (whole table) ----------------------
    for agg, word in AGGS.items():
        add("agg_salary", P_AGG_SALARY, f"SELECT {agg}(salary) FROM employees", word=word)
        add("agg_budget", P_AGG_BUDGET, f"SELECT {agg}(budget) FROM departments", word=word)

    # -- WHERE on a numeric threshold (salary), both directions -------------
    for n in SALARY_THRESHOLDS[:8]:
        add("where_salary", P_WHERE_SALARY_GT,
            f"SELECT name FROM employees WHERE salary > {n}", n=n)
        add("where_salary", P_WHERE_SALARY_LT,
            f"SELECT name FROM employees WHERE salary < {n}", n=n)

    # -- WHERE on a date (hire_date), both directions -----------------------
    for d in HIRE_DATES[:8]:
        add("where_date", P_WHERE_DATE_GT,
            f"SELECT name FROM employees WHERE hire_date > '{d}'", d=d)
        add("where_date", P_WHERE_DATE_LT,
            f"SELECT name FROM employees WHERE hire_date < '{d}'", d=d)

    # -- ORDER BY on a column, both directions ------------------------------
    for col in ["salary", "name", "hire_date"]:
        for word, direction in [("descending", "DESC"), ("ascending", "ASC")]:
            add("order_by", P_ORDER_BY,
                f"SELECT name FROM employees ORDER BY {col} {direction}",
                col=col, word=word)

    # -- COUNT(DISTINCT ...) ------------------------------------------------
    for plural, col, table in [("departments", "department", "employees"),
                               ("locations", "location", "departments")]:
        add("distinct", P_DISTINCT, f"SELECT COUNT(DISTINCT {col}) FROM {table}",
            plural=plural, table=table)

    # -- departments filtered by budget threshold, both directions ----------
    for n in BUDGET_THRESHOLDS[:8]:
        add("where_budget", P_WHERE_BUDGET_GT,
            f"SELECT * FROM departments WHERE budget > {n}", n=n)
        add("where_budget", P_WHERE_BUDGET_LT,
            f"SELECT * FROM departments WHERE budget < {n}", n=n)

    # -- department filter + ORDER BY name ----------------------------------
    for dept in DEPARTMENTS:
        add("filter_order", P_FILTER_ORDER,
            f"SELECT name FROM employees WHERE department = '{dept}' ORDER BY name",
            dept=dept)

    # -- aggregate salary within a department -------------------------------
    for dept in DEPARTMENTS:
        add("agg_in_dept", P_AGG_IN_DEPT_AVG,
            f"SELECT AVG(salary) FROM employees WHERE department = '{dept}'", dept=dept)
    for dept in DEPARTMENTS[:6]:
        add("agg_in_dept", P_AGG_IN_DEPT_MAX,
            f"SELECT MAX(salary) FROM employees WHERE department = '{dept}'", dept=dept)

    # -- SUM within a department: the contrastive partner of the "how big is
    #    the {dept} team?" count above. Both say "total"; only one means money.
    for dept in DEPARTMENTS:
        add("sum_in_dept", P_SUM_IN_DEPT,
            f"SELECT SUM(salary) FROM employees WHERE department = '{dept}'", dept=dept)

    # -- departments COUNT by location --------------------------------------
    for loc in LOCATIONS:
        add("count_location", P_COUNT_LOCATION,
            f"SELECT COUNT(*) FROM departments WHERE location = '{loc}'", loc=loc)

    # -- single extreme row (ORDER BY ... LIMIT 1) --------------------------
    add("extreme_one", P_EXTREME_MAX,
        "SELECT name FROM employees ORDER BY salary DESC LIMIT 1")
    add("extreme_one", P_EXTREME_MIN,
        "SELECT name FROM employees ORDER BY salary ASC LIMIT 1")
    add("extreme_one", P_EXTREME_RECENT,
        "SELECT name FROM employees ORDER BY hire_date DESC LIMIT 1")
    add("extreme_one", P_EXTREME_EARLIEST,
        "SELECT name FROM employees ORDER BY hire_date ASC LIMIT 1")
    add("extreme_one", P_EXTREME_DEPT_MAX,
        "SELECT name FROM departments ORDER BY budget DESC LIMIT 1")
    add("extreme_one", P_EXTREME_DEPT_MIN,
        "SELECT name FROM departments ORDER BY budget ASC LIMIT 1")

    # -- top-N (ORDER BY ... LIMIT n), both directions ----------------------
    for n in TOP_N:
        add("top_n", P_TOP_N_DESC,
            f"SELECT name FROM employees ORDER BY salary DESC LIMIT {n}", n=n)
        add("top_n", P_TOP_N_ASC,
            f"SELECT name FROM employees ORDER BY salary ASC LIMIT {n}", n=n)

    # -- GROUP BY with an aggregate (per department) ------------------------
    add("group_by", P_GROUP_COUNT,
        "SELECT department, COUNT(*) FROM employees GROUP BY department")
    for agg, phrase in [("SUM", "total salary paid"),
                        ("AVG", "average salary"),
                        ("MAX", "highest salary")]:
        add("group_by", P_GROUP_AGG,
            f"SELECT department, {agg}(salary) FROM employees GROUP BY department",
            phrase=phrase)
    add("group_by", P_GROUP_LOC_COUNT,
        "SELECT location, COUNT(*) FROM departments GROUP BY location")
    for agg, phrase in [("AVG", "average budget"), ("SUM", "total budget")]:
        add("group_by", P_GROUP_LOC_AGG,
            f"SELECT location, {agg}(budget) FROM departments GROUP BY location",
            phrase=phrase)
    # The ambiguity fix: restore "grouping column + COUNT(*) on employees" without
    # ever emitting the graded answer. Own category so the cap cannot starve it.
    for n in SALARY_THRESHOLDS[:6]:
        add("group_count_emp", P_GROUP_COUNT_FILTERED,
            f"SELECT department, COUNT(*) FROM employees WHERE salary > {n} "
            f"GROUP BY department", n=n)
    add("group_count_emp", P_GROUP_COUNT_MANAGER,
        "SELECT manager_id, COUNT(*) FROM employees GROUP BY manager_id")

    # -- GROUP BY ... HAVING, both directions -------------------------------
    for n in HAVING_N:
        add("having", P_HAVING_GT,
            f"SELECT department FROM employees GROUP BY department "
            f"HAVING COUNT(*) > {n}", n=n)
        add("having", P_HAVING_LT,
            f"SELECT department FROM employees GROUP BY department "
            f"HAVING COUNT(*) < {n}", n=n)

    # -----------------------------------------------------------------------
    # Construct families added after the first blind measurement. See the block
    # comment above P_SELECT_DISTINCT for why these exist and what the protocol
    # around them is.
    # -----------------------------------------------------------------------

    # -- SELECT DISTINCT (the "list them" half of the DISTINCT lesson) -------
    for plural, col, table in [("departments", "department", "employees"),
                               ("locations", "location", "departments")]:
        add("select_distinct", P_SELECT_DISTINCT,
            f"SELECT DISTINCT {col} FROM {table}", plural=plural, table=table)
    add("select_distinct", P_SELECT_DISTINCT_MANAGER,
        "SELECT DISTINCT manager_id FROM employees")
    for n in SALARY_THRESHOLDS[:6]:
        add("select_distinct", P_SELECT_DISTINCT_WHERE,
            f"SELECT DISTINCT department FROM employees WHERE salary > {n}", n=n)
        add("distinct", P_COUNT_DISTINCT_WHERE,
            f"SELECT COUNT(DISTINCT department) FROM employees WHERE salary > {n}",
            n=n)

    # -- IS NULL / IS NOT NULL ----------------------------------------------
    add("null_check", P_NULL_MISSING,
        "SELECT name FROM employees WHERE manager_id IS NULL")
    add("null_check", P_NULL_PRESENT,
        "SELECT name FROM employees WHERE manager_id IS NOT NULL")
    add("null_check", P_NULL_COUNT_MISSING,
        "SELECT COUNT(*) FROM employees WHERE manager_id IS NULL")
    add("null_check", P_NULL_COUNT_PRESENT,
        "SELECT COUNT(*) FROM employees WHERE manager_id IS NOT NULL")
    add("null_check", P_NULL_DEPT,
        "SELECT name FROM departments WHERE location IS NULL")
    add("null_check", P_NULL_DEPT_PRESENT,
        "SELECT * FROM departments WHERE location IS NOT NULL")

    # -- year extracted from an ISO date (strftime, not YEAR) ---------------
    for y in HIRE_YEARS:
        add("date_year", P_DATE_YEAR,
            f"SELECT name FROM employees WHERE strftime('%Y', hire_date) = '{y}'", y=y)
        add("date_year", P_DATE_YEAR_COUNT,
            f"SELECT COUNT(*) FROM employees "
            f"WHERE strftime('%Y', hire_date) = '{y}'", y=y)

    # -- LIKE: prefix, suffix and substring ---------------------------------
    for x in NAME_INITIALS:
        add("like_match", P_LIKE_PREFIX,
            f"SELECT name FROM employees WHERE name LIKE '{x}%'", x=x)
    for x in NAME_ENDINGS:
        add("like_match", P_LIKE_SUFFIX,
            f"SELECT name FROM employees WHERE name LIKE '%{x}'", x=x)
    for x in DEPT_FRAGMENTS:
        add("like_match", P_LIKE_CONTAINS,
            f"SELECT name FROM departments WHERE name LIKE '%{x}%'", x=x)

    # -- BETWEEN on a number and on a date ----------------------------------
    for lo, hi in SALARY_BANDS:
        add("between", P_BETWEEN_SALARY,
            f"SELECT name FROM employees WHERE salary BETWEEN {lo} AND {hi}",
            a=lo, b=hi)
    for lo, hi in BUDGET_BANDS:
        add("between", P_BETWEEN_BUDGET,
            f"SELECT * FROM departments WHERE budget BETWEEN {lo} AND {hi}",
            a=lo, b=hi)
    for lo, hi in DATE_BANDS:
        add("between", P_BETWEEN_DATE,
            f"SELECT name FROM employees "
            f"WHERE hire_date BETWEEN '{lo}' AND '{hi}'", a=lo, b=hi)

    # -- negation (!=) -------------------------------------------------------
    for dept in DEPARTMENTS[:8]:
        add("not_equal", P_NOT_EQUAL_DEPT,
            f"SELECT name FROM employees WHERE department != '{dept}'", dept=dept)
    for loc in LOCATIONS[:8]:
        add("not_equal", P_NOT_EQUAL_LOC,
            f"SELECT * FROM departments WHERE location != '{loc}'", loc=loc)

    # -----------------------------------------------------------------------
    # Multi-table JOINs. Parameter pools are deliberately narrower here than
    # above: each pattern is capped at MAX_PER_CATEGORY anyway, so trading
    # literal breadth for phrasing depth means every join target is still asked
    # several ways after balancing (see balance_categories).
    # -----------------------------------------------------------------------

    # -- project columns from both tables -----------------------------------
    for ecol in ["name", "salary", "hire_date"]:
        for dcol in ["budget", "location"]:
            add("join_project", P_JOIN_PROJECT,
                f"SELECT employees.{ecol}, departments.{dcol} {JOIN_DEPT}",
                ecol=ecol.replace("_", " "), dcol=dcol)

    # -- filter on a column that only exists in the joined table ------------
    for loc in LOCATIONS[:4]:
        add("join_where", P_JOIN_WHERE_LOC,
            f"SELECT employees.name {JOIN_DEPT} WHERE departments.location = '{loc}'",
            loc=loc)
    for n in BUDGET_THRESHOLDS[:2]:
        add("join_where", P_JOIN_WHERE_BUDGET_GT,
            f"SELECT employees.name {JOIN_DEPT} WHERE departments.budget > {n}", n=n)
        add("join_where", P_JOIN_WHERE_BUDGET_LT,
            f"SELECT employees.name {JOIN_DEPT} WHERE departments.budget < {n}", n=n)

    # -- aggregate over a join, filtered by the joined table ----------------
    for loc in LOCATIONS[:4]:
        add("join_count", P_JOIN_COUNT_LOC,
            f"SELECT COUNT(*) {JOIN_DEPT} WHERE departments.location = '{loc}'", loc=loc)
    for agg, word in AGGS.items():
        for loc in LOCATIONS[:2]:
            add("join_count", P_JOIN_AGG_LOC,
                f"SELECT {agg}(employees.salary) {JOIN_DEPT} "
                f"WHERE departments.location = '{loc}'", word=word, loc=loc)

    # -- GROUP BY a column of the joined table (impossible without the join) --
    add("join_group", P_JOIN_GROUP_COUNT,
        f"SELECT departments.location, COUNT(*) {JOIN_DEPT} GROUP BY departments.location")
    for agg, phrase in [("AVG", "average salary"),
                        ("SUM", "total salary paid"),
                        ("MAX", "highest salary")]:
        add("join_group", P_JOIN_GROUP_AGG,
            f"SELECT departments.location, {agg}(employees.salary) {JOIN_DEPT} "
            f"GROUP BY departments.location", phrase=phrase)
    for n in HAVING_N[2:]:
        add("join_group", P_JOIN_HAVING_GT,
            f"SELECT departments.location {JOIN_DEPT} GROUP BY departments.location "
            f"HAVING COUNT(*) > {n}", n=n)
        add("join_group", P_JOIN_HAVING_LT,
            f"SELECT departments.location {JOIN_DEPT} GROUP BY departments.location "
            f"HAVING COUNT(*) < {n}", n=n)

    # -- ORDER BY / LIMIT over a filtered join ------------------------------
    for loc in LOCATIONS[:3]:
        for word, direction in [("descending", "DESC"), ("ascending", "ASC")]:
            add("join_order", P_JOIN_ORDER,
                f"SELECT employees.name {JOIN_DEPT} "
                f"WHERE departments.location = '{loc}' "
                f"ORDER BY employees.salary {direction}", loc=loc, word=word)
    for loc in LOCATIONS[:4]:
        add("join_order", P_JOIN_TOP1,
            f"SELECT employees.name {JOIN_DEPT} WHERE departments.location = '{loc}' "
            f"ORDER BY employees.salary DESC LIMIT 1", loc=loc)

    # -- self-join: a row of employees related to another row of employees ---
    for mcol in ["name", "salary", "department", "hire_date"]:
        add("self_join", P_SELF_JOIN_PROJECT,
            f"SELECT e.name, m.{mcol} {JOIN_SELF}", mcol=mcol.replace("_", " "))
    add("self_join", P_SELF_JOIN_COUNT, f"SELECT COUNT(*) {JOIN_SELF}")
    add("self_join", P_SELF_JOIN_DISTINCT, f"SELECT DISTINCT m.name {JOIN_SELF}")
    for dept in DEPARTMENTS[:4]:
        add("self_join", P_SELF_JOIN_WHERE,
            f"SELECT e.name {JOIN_SELF} WHERE m.department = '{dept}'", dept=dept)

    return out


def normalize_question(q: str) -> str:
    """Lightweight question canonicaliser for dedup + leakage checks."""
    return " ".join(q.lower().split()).rstrip("?.").strip()


def question_words(q: str) -> frozenset:
    """Bag of words used for the near-duplicate (overlap) check."""
    return frozenset(normalize_question(q).replace("?", " ").replace(",", " ").split())


def jaccard(a: frozenset, b: frozenset) -> float:
    """Word-level Jaccard similarity; 1.0 means identical wording."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def closest_eval_question(q: str, eval_word_sets: List[Tuple[frozenset, str]]) -> Tuple[float, str]:
    """Highest word overlap between this question and any eval question."""
    words = question_words(q)
    best, best_q = 0.0, ""
    for ev_words, ev_q in eval_word_sets:
        score = jaccard(words, ev_words)
        if score > best:
            best, best_q = score, ev_q
    return best, best_q


def balance_categories(kept: List[Example], cap: int, rng: random.Random) -> Tuple[List[Example], int]:
    """Cap each pattern at `cap` examples, keeping phrasing + parameter spread.

    Sampling round-robin over SQL targets (rather than taking a random slice)
    keeps every parameter value represented before any target contributes a
    second phrasing, so a capped pattern still shows several wordings.
    """
    by_cat: Dict[str, List[Example]] = defaultdict(list)
    for ex in kept:
        by_cat[ex[0]].append(ex)

    balanced: List[Example] = []
    dropped = 0
    for cat in sorted(by_cat):
        items = by_cat[cat]
        if len(items) <= cap:
            balanced.extend(items)
            continue
        by_sql: Dict[str, List[Example]] = defaultdict(list)
        for ex in items:
            by_sql[normalize_sql(ex[2])].append(ex)
        for group in by_sql.values():
            rng.shuffle(group)
        targets = sorted(by_sql)
        rng.shuffle(targets)
        picked: List[Example] = []
        depth = 0
        while len(picked) < cap:
            progressed = False
            for sql in targets:
                if depth < len(by_sql[sql]):
                    picked.append(by_sql[sql][depth])
                    progressed = True
                    if len(picked) == cap:
                        break
            if not progressed:
                break
            depth += 1
        balanced.extend(picked)
        dropped += len(items) - len(picked)
    return balanced, dropped


def build(eval_files: List[Path], val_frac: float, seed: int,
          cap: int = MAX_PER_CATEGORY) -> Dict[str, object]:
    """Generate, dedup, de-leak, balance, and stratified-split. Returns a report."""
    rng = random.Random(seed)

    # Leakage blocklists from EVERY held-out eval set (in-template, paraphrase,
    # cross-schema), so a paraphrased training question cannot collide with the
    # very set used to measure robustness to paraphrase.
    eval_rows: List[Dict[str, str]] = []
    for path in eval_files:
        eval_rows.extend(load_jsonl(path))
    eval_questions = {normalize_question(r["question"]) for r in eval_rows}
    eval_sqls = {normalize_sql(r["sql"]) for r in eval_rows}
    eval_word_sets = [(question_words(r["question"]), r["question"]) for r in eval_rows]

    candidates = generate_candidates()

    kept: List[Example] = []
    seen_questions: set = set()
    dropped_leak = 0
    dropped_dup = 0
    worst_overlap, worst_pair = 0.0, ("", "")
    for cat, q, sql in candidates:
        nq, nsql = normalize_question(q), normalize_sql(sql)
        if nq in eval_questions or nsql in eval_sqls:   # rule 2: no leakage
            dropped_leak += 1
            continue
        if nq in seen_questions:                        # de-duplicate questions
            dropped_dup += 1
            continue
        overlap, near = closest_eval_question(q, eval_word_sets)
        if overlap > worst_overlap:                     # rule 3: measure closeness
            worst_overlap, worst_pair = overlap, (q, near)
        seen_questions.add(nq)
        kept.append((cat, q, sql))

    # Rule 4: cap each pattern so the biggest parameter pools cannot dominate.
    n_before_balance = len(kept)
    kept, dropped_cap = balance_categories(kept, cap, rng)

    # Stratified split: hold out ~val_frac of EACH category so validation
    # mirrors every SQL pattern (otherwise a rare pattern might be train-only).
    by_cat: Dict[str, List[Example]] = defaultdict(list)
    for ex in kept:
        by_cat[ex[0]].append(ex)

    train: List[Example] = []
    val: List[Example] = []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        n_val = min(len(items) - 1, math.ceil(len(items) * val_frac)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    # Diversity stat: how many different questions map to the same SQL target.
    per_sql: Dict[str, int] = defaultdict(int)
    for _cat, q, sql in kept:
        per_sql[normalize_sql(sql)] += 1
    phrasings_per_sql = sum(per_sql.values()) / max(len(per_sql), 1)

    return {
        "train": train,
        "val": val,
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "dropped_leak": dropped_leak,
        "dropped_dup": dropped_dup,
        "dropped_cap": dropped_cap,
        "n_before_balance": n_before_balance,
        "cap": cap,
        "worst_overlap": worst_overlap,
        "worst_pair": worst_pair,
        "n_eval_rows": len(eval_rows),
        "n_sql_targets": len(per_sql),
        "phrasings_per_sql": phrasings_per_sql,
        "by_cat": {c: len(v) for c, v in sorted(by_cat.items())},
        "eval_questions": eval_questions,
        "eval_sqls": eval_sqls,
        "eval_word_sets": eval_word_sets,
    }


def write_jsonl(rows: List[Example], path: Path) -> None:
    """Write clean {id, question, sql} records (schema identical to eval)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, (_cat, q, sql) in enumerate(rows, start=1):
            fh.write(json.dumps({"id": i, "question": q, "sql": sql}) + "\n")


def verify_no_leakage(rows: List[Example], eval_questions: set, eval_sqls: set,
                      eval_word_sets: List[Tuple[frozenset, str]]) -> None:
    """Assert (belt and braces) that nothing written collides with eval."""
    for _cat, q, sql in rows:
        assert normalize_question(q) not in eval_questions, f"LEAK question: {q}"
        assert normalize_sql(sql) not in eval_sqls, f"LEAK sql: {sql}"
        overlap, near = closest_eval_question(q, eval_word_sets)
        assert overlap < IDENTICAL_OVERLAP, f"LEAK near-duplicate: {q!r} vs {near!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the NL->SQL training set.")
    parser.add_argument("--eval-file", nargs="+", default=None,
                        help="eval sets to de-leak against (default: every data/eval/*.jsonl)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-per-category", type=int, default=MAX_PER_CATEGORY,
                        help="cap examples contributed by any one SQL pattern")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    eval_files = [Path(p) for p in args.eval_file] if args.eval_file else default_eval_files()
    if not eval_files:
        print(f"no eval sets found in {DEFAULT_EVAL_DIR}", file=sys.stderr)
        return 1

    rep = build(eval_files, args.val_frac, args.seed, cap=args.max_per_category)
    train, val = rep["train"], rep["val"]

    verify_no_leakage(train + val, rep["eval_questions"], rep["eval_sqls"],
                      rep["eval_word_sets"])

    out_dir = Path(args.out_dir)
    train_path = out_dir / "text2sql_train.jsonl"
    val_path = out_dir / "text2sql_val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print("=" * 64)
    print("PromptQL training-set build")
    print("=" * 64)
    print("de-leaked against    : "
          + ", ".join(p.name for p in eval_files)
          + f"  ({rep['n_eval_rows']} questions)")
    print(f"candidates generated : {rep['n_candidates']}")
    print(f"dropped (eval leak)  : {rep['dropped_leak']}")
    print(f"dropped (duplicate)  : {rep['dropped_dup']}")
    print(f"dropped (cat cap)    : {rep['dropped_cap']} "
          f"(max {rep['cap']} per pattern, from {rep['n_before_balance']})")
    print(f"kept (unique, clean) : {rep['n_kept']}")
    print(f"  -> train           : {len(train)}")
    print(f"  -> val             : {len(val)}")
    print(f"distinct SQL targets : {rep['n_sql_targets']} "
          f"({rep['phrasings_per_sql']:.1f} phrasings each on average)")
    print("leakage vs eval      : 0 (verified)")
    print(f"closest eval question: {rep['worst_overlap']:.0%} word overlap "
          f"(differs by a literal; its SQL target is not in any eval set)")
    if rep["worst_pair"][0]:
        print(f"   train: {rep['worst_pair'][0]}")
        print(f"   eval : {rep['worst_pair'][1]}")
    print("-" * 64)
    print("per-pattern (category) counts among kept examples:")
    train_cats = Counter(c for c, _, _ in train)
    val_cats = Counter(c for c, _, _ in val)
    for cat in sorted(rep["by_cat"]):
        print(f"  {cat:16s} total={rep['by_cat'][cat]:3d}  "
              f"train={train_cats.get(cat,0):3d}  val={val_cats.get(cat,0):2d}")
    print("-" * 64)
    print(f"wrote {train_path.relative_to(REPO_ROOT)}")
    print(f"wrote {val_path.relative_to(REPO_ROOT)}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
