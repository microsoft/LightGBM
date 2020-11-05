import json
from os import environ
from sys import argv, exit
from time import sleep
try:
    from urllib import request
except ImportError:
    import urllib2 as request


def get_runs(workflow_id):
    pr_runs = []
    if environ.get("GITHUB_EVENT_NAME", "") == "pull_request":
        req = request.Request(url="{}/repos/{}/actions/workflows/{}/runs".format(environ.get("GITHUB_API_URL"),
                                                                                 environ.get("GITHUB_REPOSITORY"),
                                                                                 workflow_id),
                              headers={"Accept": "application/vnd.github.v3+json"})
        url = request.urlopen(req)
        data = json.loads(url.read().decode('utf-8'))
        url.close()
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
            if run['conclusion'] in {'failure', 'timed_out', 'cancelled'}:
                status = 'fail'
                break
            if run['conclusion'] == 'success':
                status = 'ok'
                break
        if run['status'] in {'in_progress', 'queued'}:
            status = 'rerun'
            break
    return status


if __name__ == "__main__":
    workflow_id = argv[1]
    while True:
        status = get_status(get_runs(workflow_id))
        if status != 'rerun':
            break
        sleep(60)
    if status == 'fail':
        exit(1)
