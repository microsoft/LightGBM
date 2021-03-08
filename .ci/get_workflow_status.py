# coding: utf-8
"""Get the most recent status of workflow for the current PR.

[usage]
    python get_workflow_status.py TRIGGER_PHRASE

TRIGGER_PHRASE: Code phrase that triggers workflow.
"""
import json
from os import environ
from sys import argv, exit
from time import sleep

try:
    from urllib import request
except ImportError:
    import urllib2 as request


def get_runs(trigger_phrase):
    """Get all triggering workflow comments in the current PR.

    Parameters
    ----------
    trigger_phrase : string
        Code phrase that triggers workflow.

    Returns
    -------
    pr_runs : list
        List of comment objects sorted by the time of creation in decreasing order.
    """
    pr_runs = []
    if environ.get("GITHUB_EVENT_NAME", "") == "pull_request":
        pr_number = int(environ.get("GITHUB_REF").split('/')[-2])
        req = request.Request(url="{}/repos/microsoft/LightGBM/issues/{}/comments".format(environ.get("GITHUB_API_URL"),
                                                                                          pr_number),
                              headers={"Accept": "application/vnd.github.v3+json"})
        url = request.urlopen(req)
        data = json.loads(url.read().decode('utf-8'))
        url.close()
        pr_runs = [i for i in data
                   if i['author_association'].lower() in {'owner', 'member', 'collaborator'}
                   and i['body'].startswith('/gha run {}'.format(trigger_phrase))]
    return pr_runs[::-1]


def get_status(runs):
    """Get the most recent status of workflow for the current PR.

    Parameters
    ----------
    runs : list
        List of comment objects sorted by the time of creation in decreasing order.

    Returns
    -------
    status : string
        The most recent status of workflow.
        Can be 'success', 'failure' or 'in-progress'.
    """
    status = 'success'
    for run in runs:
        body = run['body']
        if "Status: " in body:
            if "Status: skipped" in body:
                continue
            if "Status: failure" in body:
                status = 'failure'
                break
            if "Status: success" in body:
                status = 'success'
                break
        else:
            status = 'in-progress'
            break
    return status


if __name__ == "__main__":
    trigger_phrase = argv[1]
    while True:
        status = get_status(get_runs(trigger_phrase))
        if status != 'in-progress':
            break
        sleep(60)
    if status == 'failure':
        exit(1)
