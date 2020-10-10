import json
from os import environ
from sys import exit
from time import sleep
from urllib import request


def get_runs():
    with request.urlopen("https://api.github.com/repos/microsoft/LightGBM/actions/workflows/test_1.yml/runs") as url:
        data = json.loads(url.read().decode())
    pr_runs = []
    if environ.get("GITHUB_EVENT_NAME", "") == "pull_request":
        pr_runs = [i for i in data['workflow_runs']
                   if i['event'] == 'pull_request_review_comment' and
                   (i.get('pull_requests') and
                    i['pull_requests'][0]['number'] == int(environ.get("GITHUB_REF").split('/')[-2]) or
                    i['head_branch'] == environ.get("GITHUB_HEAD_REF").split('/')[-1])]
    return sorted(pr_runs, key=lambda i: i['run_number'], reverse=True)


def get_status(runs):
    status = 'ok'
    for run in runs:
        if run['status'] == 'completed':
            if run['conclusion'] == 'skipped':
                continue
            if run['conclusion'] == 'failure':
                status = 'fail'
                break
            if run['conclusion'] == 'success':
                break
        if run['status'] in {'in_progress', 'queued'}:
            status = 'rerun'
            break
    return status


if __name__ == "__main__":
    while True:
        status = get_status(get_runs())
        if status != 'rerun':
            break
        sleep(60)
    if status == 'fail':
        exit(1)
