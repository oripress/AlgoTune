#!/usr/bin/env python3

import os
from pathlib import Path

import boto3
from dotenv import load_dotenv


# Load environment from both .env files
# 1. Load API keys from root .env
root_dotenv = Path(__file__).parent.parent / ".env"
if root_dotenv.exists():
    load_dotenv(root_dotenv)

# 2. Load AWS configuration from aws/.env
aws_dotenv = Path(__file__).parent / ".env"
load_dotenv(aws_dotenv)

region = os.getenv("AWS_REGION")
batch = boto3.client("batch", region_name=region)
ec2 = boto3.client("ec2", region_name=region)

# Your AWS resource names
CE_NAME = os.getenv("BATCH_COMPUTE_ENV_NAME", "AlgoTuneCE")
QUEUE_NAME = os.getenv("BATCH_JOB_QUEUE_NAME", "AlgoTuneQueue")
JOB_DEF_NAME = os.getenv("BATCH_JOB_DEF_NAME", "AlgoTuneJobDef")

# Docker image you built & pushed (DockerHub)
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "algotune:latest")

# Optional: launch template for root volume sizing
root_volume_gb = os.getenv("BATCH_ROOT_VOLUME_GB")
launch_template_name = (
    os.getenv("BATCH_LAUNCH_TEMPLATE_NAME", f"{CE_NAME}-lt") if root_volume_gb else None
)

# Roles - get account ID to construct ARNs
account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
ecs_instance_role_name = os.getenv("ECS_INSTANCE_ROLE", "ecsInstanceRole")
ecs_task_exec_role_name = os.getenv("ECS_TASK_EXEC_ROLE", "ecsTaskExecutionRole")

# Convert instance role name to instance profile ARN if not already an ARN
if ecs_instance_role_name.startswith("arn:"):
    ecs_instance_role = ecs_instance_role_name
else:
    # AWS Batch needs the instance PROFILE ARN, not the role ARN
    ecs_instance_role = f"arn:aws:iam::{account_id}:instance-profile/{ecs_instance_role_name}"

# Convert task execution role name to ARN if not already an ARN
if ecs_task_exec_role_name.startswith("arn:"):
    ecs_task_exec_role = ecs_task_exec_role_name
else:
    ecs_task_exec_role = f"arn:aws:iam::{account_id}:role/{ecs_task_exec_role_name}"

batch_service_role = os.getenv("AWS_BATCH_SERVICE_ROLE")  # will be auto-created once

# Network config
subnets = os.getenv("AWS_SUBNET_ID", "").split(",")
security_groups = os.getenv("AWS_SG_ID", "").split(",")


# Ensure launch template exists if root volume override is requested.
def ensure_launch_template():
    if not root_volume_gb:
        return None
    volume_size = int(root_volume_gb)
    lt_data = {
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": volume_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ]
    }

    # Try to create/update the launch template
    # Due to IAM permissions, we might not be able to check if it exists
    # Try to create a new version first (if template exists), otherwise create new
    try:
        ec2.create_launch_template_version(
            LaunchTemplateName=launch_template_name,
            LaunchTemplateData=lt_data,
        )
        print(f"  ✓ Updated launch template '{launch_template_name}' (root {volume_size}GB)")
    except ec2.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if (
            "InvalidLaunchTemplateName.NotFoundException" in error_code
            or "NotFoundException" in error_code
        ):
            # Template doesn't exist, create it
            try:
                ec2.create_launch_template(
                    LaunchTemplateName=launch_template_name,
                    LaunchTemplateData=lt_data,
                    VersionDescription=f"root {volume_size}GB",
                )
                print(
                    f"  ✓ Created launch template '{launch_template_name}' (root {volume_size}GB)"
                )
            except ec2.exceptions.ClientError as e2:
                error_code2 = e2.response.get("Error", {}).get("Code", "")
                if "UnauthorizedOperation" in error_code2:
                    print("  ⚠ Warning: No EC2 permissions to manage launch templates")
                    print(f"    Will attempt to use existing template '{launch_template_name}'")
                else:
                    raise
        elif "UnauthorizedOperation" in error_code:
            print("  ⚠ Warning: No EC2 permissions to manage launch templates")
            print(f"    Will attempt to use existing template '{launch_template_name}'")
        else:
            raise

    return launch_template_name


if root_volume_gb:
    ensure_launch_template()

# Check if compute environment already exists
print(f"Checking if compute environment '{CE_NAME}' exists...")
try:
    resp = batch.describe_compute_environments(computeEnvironments=[CE_NAME])
    if resp["computeEnvironments"]:
        print(f"  ✓ Compute environment '{CE_NAME}' already exists")
        ce_exists = True
        existing_ce = resp["computeEnvironments"][0]
    else:
        ce_exists = False
        existing_ce = None
except Exception:
    ce_exists = False
    existing_ce = None

if not ce_exists:
    print(f"Creating compute environment '{CE_NAME}' ...")

    # Build the compute environment config
    min_vcpus = int(os.getenv("BATCH_MIN_VCPUS", "0"))
    max_vcpus = int(os.getenv("BATCH_MAX_VCPUS", "64"))
    desired_vcpus = int(os.getenv("BATCH_DESIRED_VCPUS", str(min_vcpus)))

    ce_config = {
        "computeEnvironmentName": CE_NAME,
        "type": "MANAGED",
        "state": "ENABLED",
        "computeResources": {
            "type": "EC2",
            "minvCpus": min_vcpus,
            "maxvCpus": max_vcpus,
            "desiredvCpus": desired_vcpus,
            "instanceTypes": os.getenv("BATCH_INSTANCE_TYPES", "c6a.4xlarge").split(","),
            "subnets": subnets,
            "securityGroupIds": security_groups,
            "instanceRole": ecs_instance_role,
        },
    }
    if root_volume_gb:
        ce_config["computeResources"]["launchTemplate"] = {
            "launchTemplateName": launch_template_name,
            "version": "$Latest",
        }

    # Only add serviceRole if it's not a service-linked role (service-linked roles are auto-used)
    if batch_service_role and "aws-service-role" not in batch_service_role:
        ce_config["serviceRole"] = batch_service_role

    resp_ce = batch.create_compute_environment(**ce_config)
    print("Compute Environment creation request submitted:", resp_ce)
else:
    # Compute environment exists - check if it needs to be re-enabled/scaled up
    # (e.g., after Ctrl+C kill switch scaled it down)
    min_vcpus = int(os.getenv("BATCH_MIN_VCPUS", "0"))
    max_vcpus = int(os.getenv("BATCH_MAX_VCPUS", "64"))
    desired_vcpus = int(os.getenv("BATCH_DESIRED_VCPUS", str(min_vcpus)))

    current_state = existing_ce.get("state")
    current_resources = existing_ce.get("computeResources", {})
    current_min = current_resources.get("minvCpus")
    current_max = current_resources.get("maxvCpus")
    current_desired = current_resources.get("desiredvCpus")
    current_lt = current_resources.get("launchTemplate", {})

    needs_update = False
    update_params = {"computeEnvironment": CE_NAME}

    # Re-enable if disabled (e.g., by Ctrl+C kill switch)
    if current_state == "DISABLED":
        print("  → Compute environment is DISABLED, re-enabling...")
        update_params["state"] = "ENABLED"
        needs_update = True

    # Ensure compute resources match current run settings (keeps ASG in sync).
    if current_min != min_vcpus or current_max != max_vcpus or current_desired != desired_vcpus:
        print(
            "  → Updating compute resources to "
            f"min={min_vcpus}, max={max_vcpus}, desired={desired_vcpus}..."
        )
        update_params["computeResources"] = {
            "minvCpus": min_vcpus,
            "maxvCpus": max_vcpus,
            "desiredvCpus": desired_vcpus,
        }
        needs_update = True
    if root_volume_gb:
        desired_lt = {
            "launchTemplateName": launch_template_name,
            "version": "$Latest",
        }
        if current_lt != desired_lt:
            update_params.setdefault("computeResources", {})
            update_params["computeResources"]["launchTemplate"] = desired_lt
            needs_update = True

    if needs_update:
        try:
            batch.update_compute_environment(**update_params)
            print("  ✓ Compute environment updated")
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to update compute environment: {e}")

print(f"Waiting for compute environment '{CE_NAME}' to become VALID...")
import time


max_wait = 300  # 5 minutes
wait_interval = 10
elapsed = 0

while elapsed < max_wait:
    resp = batch.describe_compute_environments(computeEnvironments=[CE_NAME])
    if resp["computeEnvironments"]:
        status = resp["computeEnvironments"][0]["status"]
        if status == "VALID":
            print("  ✓ Compute environment is VALID")
            break
        elif status == "INVALID":
            reason = resp["computeEnvironments"][0].get("statusReason", "Unknown")
            print(f"  ✗ Compute environment is INVALID: {reason}")
            raise Exception(f"Compute environment failed: {reason}")
        else:
            print(f"  Status: {status}, waiting... ({elapsed}s elapsed)")
    time.sleep(wait_interval)
    elapsed += wait_interval

if elapsed >= max_wait:
    print("  ⚠️  Timeout waiting for compute environment (status may still be creating)")
    print("     Continuing anyway - queue creation may fail")

# Check if job queue already exists
print(f"Checking if job queue '{QUEUE_NAME}' exists...")
try:
    resp = batch.describe_job_queues(jobQueues=[QUEUE_NAME])
    if resp["jobQueues"]:
        print(f"  ✓ Job queue '{QUEUE_NAME}' already exists")
        queue_exists = True
        existing_queue = resp["jobQueues"][0]
    else:
        queue_exists = False
        existing_queue = None
except Exception:
    queue_exists = False
    existing_queue = None

if not queue_exists:
    print(f"Creating job queue '{QUEUE_NAME}' ...")
    resp_queue = batch.create_job_queue(
        jobQueueName=QUEUE_NAME,
        state="ENABLED",
        priority=int(os.getenv("BATCH_QUEUE_PRIORITY", "1")),
        computeEnvironmentOrder=[{"order": 1, "computeEnvironment": CE_NAME}],
    )
    print("Job queue created:", resp_queue)
else:
    desired_order = [{"order": 1, "computeEnvironment": CE_NAME}]
    existing_order = existing_queue.get("computeEnvironmentOrder", []) if existing_queue else []
    desired_priority = int(os.getenv("BATCH_QUEUE_PRIORITY", "1"))
    existing_priority = existing_queue.get("priority") if existing_queue else None
    existing_state = existing_queue.get("state") if existing_queue else None
    if (
        existing_order != desired_order
        or existing_priority != desired_priority
        or existing_state != "ENABLED"
    ):
        print(f"Updating job queue '{QUEUE_NAME}' to use compute environment '{CE_NAME}' ...")
        resp_queue = batch.update_job_queue(
            jobQueue=QUEUE_NAME,
            state="ENABLED",
            priority=desired_priority,
            computeEnvironmentOrder=desired_order,
        )
        print("Job queue updated:", resp_queue)
    else:
        print(f"  ✓ Job queue '{QUEUE_NAME}' already configured for '{CE_NAME}'")

# Check if job definition already exists and is ACTIVE
print(f"Checking if job definition '{JOB_DEF_NAME}' exists...")
try:
    resp = batch.describe_job_definitions(jobDefinitionName=JOB_DEF_NAME, status="ACTIVE")
    if resp["jobDefinitions"]:
        print(f"  ✓ Job definition '{JOB_DEF_NAME}' already exists")
        jobdef_exists = True
    else:
        jobdef_exists = False
except Exception:
    jobdef_exists = False

if jobdef_exists:
    print(
        f"Registering new revision for job definition '{JOB_DEF_NAME}' to apply current settings..."
    )
else:
    print(f"Registering job definition '{JOB_DEF_NAME}' ...")

# Task role for container AWS API access (S3, etc.)
task_role_arn = f"arn:aws:iam::{account_id}:role/AlgoTuneBatchTaskRole"

resp_jobdef = batch.register_job_definition(
    jobDefinitionName=JOB_DEF_NAME,
    type="container",
    timeout={
        "attemptDurationSeconds": 432000  # 5 days (5 * 24 * 60 * 60)
    },
    containerProperties={
        "image": DOCKER_IMAGE,
        "executionRoleArn": ecs_task_exec_role,
        "jobRoleArn": task_role_arn,  # Container's AWS API access role
        "vcpus": int(os.getenv("BATCH_JOB_VCPUS", "1")),
        "memory": int(os.getenv("BATCH_JOB_MEMORY_MB", "16000")),
        "command": [],  # we'll override at submission
        # Note: Using default CloudWatch logging + S3 uploads every 60s
    },
)
print("Job definition registered:", resp_jobdef)

print("🎉 AWS Batch compute environment, job queue, and job definition are ready!")
