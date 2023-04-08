"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""

import copy
import queue
import string
import random
import collections


def standardize_variables(nonstandard_rules):
    """
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).

    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    """
    variables = set()

    def _replace_something(proposition, unique_name):
        if "something" in proposition:
            variables.add(unique_name)
        return [word if word != "something" else unique_name for word in proposition]

    def _standardize(rule, rule_name):
        standardized_rule = copy.deepcopy(rule)
        unique_name = str(rule_name) + "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )

        standardized_rule["antecedents"] = [
            _replace_something(antecedent, unique_name)
            for antecedent in standardized_rule["antecedents"]
        ]
        standardized_rule["consequent"] = _replace_something(
            standardized_rule["consequent"], unique_name
        )
        return standardized_rule

    standardized_rules = {
        rule_name: _standardize(rule, rule_name)
        for rule_name, rule in nonstandard_rules.items()
    }
    return standardized_rules, list(variables)


def unify(query, datum, variables):
    """
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.

    Unification succeeds if (1) every variable x in the unified query is replaced by a
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is
      detected, the query is changed to ['bobcat','eats','bobcat',True], which
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the
      rest of the contents of the query or datum.
    """
    if query[-1] != datum[-1] or len(query) != len(datum):
        return None, None

    querycp = copy.deepcopy(query)
    datumcp = copy.deepcopy(datum)
    subs = {}
    for i in range(len(querycp) - 1):
        qword = querycp[i]
        dword = datumcp[i]

        if qword == dword:
            continue
        if qword not in variables and dword not in variables:
            return None, None

        kept, substituted = (qword, dword) if qword not in variables else (dword, qword)
        querycp = [kept if word == substituted else word for word in querycp]
        datumcp = [kept if word == substituted else word for word in datumcp]
        subs[substituted] = kept

    return querycp, subs


def apply(rule, goals, variables):
    """
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.

    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).
       If every one of the goals can be unified with the rule consequent, then
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with
       applications[i]['consequent'] has been removed, and replaced by
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    """
    applications = []

    def _apply_subs(subs, proposition, variables):
        return [word if word not in subs else subs[word] for word in proposition]

    goalsets = []
    for idx, goal in enumerate(goals):
        unification, subs = unify(goal, rule["consequent"], variables)
        if not unification:
            continue

        application = copy.deepcopy(rule)
        application["antecedents"] = [
            _apply_subs(subs, antecedent, variables)
            for antecedent in application["antecedents"]
        ]
        application["consequent"] = _apply_subs(
            subs, application["consequent"], variables
        )
        applications.append(application)

        newgoals = copy.deepcopy(goals)
        newgoals.pop(idx)
        newgoals.extend(application["antecedents"])
        goalsets.append(newgoals)

    return applications, goalsets


State = collections.namedtuple("State", ["applist", "goalset"])


def backward_chain(query, rules, variables):
    """
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    """
    frontiers = queue.Queue()
    frontiers.put(State([], [query]))

    while not frontiers.empty():
        pivot = frontiers.get()

        # We have proven the query if the goalset is empty
        if pivot.goalset == []:
            return pivot.applist

        # Else we expand the pivot
        for rule_name, rule in rules.items():
            applications, new_goalsets = apply(rule, pivot.goalset, variables)
            for application, new_goalset in zip(applications, new_goalsets):
                frontiers.put(State(pivot.applist + [application], new_goalset))

    return None
