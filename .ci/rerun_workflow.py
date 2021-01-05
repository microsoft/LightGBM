import json
from os import environ
from sys import argv, exit
try:
    from urllib import request
except ImportError:
    import urllib2 as request


def get_runs(pr_number, pr_branch, workflow_id):
    req = request.Request(url="{}/repos/microsoft/LightGBM/actions/workflows/{}/runs?"
                              "event=pull_request&branch={}".format(environ.get("GITHUB_API_URL"),
                                                                    workflow_id,
                                                                    pr_branch),
                          headers={"accept": "application/vnd.github.v3+json"})
    url = request.urlopen(req)
    data = json.loads(url.read().decode('utf-8'))
    url.close()
    pr_runs = [i for i in data['workflow_runs']
               if i['event'] == 'pull_request' and
               (i.get('pull_requests') and
                i['pull_requests'][0]['number'] == int(pr_number) or
                i['head_branch'] == pr_branch)]
    return sorted(pr_runs, key=lambda i: i['run_number'], reverse=True)


def rerun_workflow(runs):
    if runs:
        req = request.Request(url="{}/repos/microsoft/LightGBM/actions/runs/{}/rerun".format(environ.get("GITHUB_API_URL"),
                                                                                             runs[0]["id"]),
                              headers={"accept": "application/vnd.github.v3+json",
                                       "authorization": "Token {}".format(environ.get("SECRETS_WORKFLOW"))},
                              data={})
        try:
            res = request.urlopen(req)
            res.close()
            if res.getcode() != 201:
                raise Exception("Cannot rerun workflow. HTTP status code: {}.".format(res.getcode()))
        except BaseException as e:
            print(e)
            exit(1)


if __name__ == "__main__":
    workflow_id = argv[1]
    pr_number = argv[2]
    pr_branch = argv[3]
    rerun_workflow(get_runs(pr_number, pr_branch, workflow_id))
