#!/usr/bin/env python3

import os
import boto3
from pathlib import Path
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

# Your AWS resource names
CE_NAME = os.getenv("BATCH_COMPUTE_ENV_NAME", "AlgoTuneCE")
QUEUE_NAME = os.getenv("BATCH_JOB_QUEUE_NAME", "AlgoTuneQueue")
JOB_DEF_NAME = os.getenv("BATCH_JOB_DEF_NAME", "AlgoTuneJobDef")

# Docker image you built & pushed (DockerHub)
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "algotune:latest")

# Roles - get account ID to construct ARNs
account_id = boto3.client('sts', region_name=region).get_caller_identity()['Account']
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

# Check if compute environment already exists
print(f"Checking if compute environment '{CE_NAME}' exists...")
try:
    resp = batch.describe_compute_environments(computeEnvironments=[CE_NAME])
    if resp['computeEnvironments']:
        print(f"  ✓ Compute environment '{CE_NAME}' already exists")
        ce_exists = True
    else:
        ce_exists = False
except Exception as e:
    ce_exists = False

if not ce_exists:
    print(f"Creating compute environment '{CE_NAME}' ...")

    # Build the compute environment config
    ce_config = {
        "computeEnvironmentName": CE_NAME,
        "type": "MANAGED",
        "state": "ENABLED",
        "computeResources": {
            "type": "EC2",
            "minvCpus": 0,
            "maxvCpus": int(os.getenv("BATCH_MAX_VCPUS", "64")),
            "desiredvCpus": 0,
            "instanceTypes": os.getenv("BATCH_INSTANCE_TYPES", "c6a.4xlarge").split(","),
            "subnets": subnets,
            "securityGroupIds": security_groups,
            "instanceRole": ecs_instance_role,
        }
    }

    # Only add serviceRole if it's not a service-linked role (service-linked roles are auto-used)
    if batch_service_role and "aws-service-role" not in batch_service_role:
        ce_config["serviceRole"] = batch_service_role

    resp_ce = batch.create_compute_environment(**ce_config)
    print("Compute Environment creation request submitted:", resp_ce)

print(f"Waiting for compute environment '{CE_NAME}' to become VALID...")
import time
max_wait = 300  # 5 minutes
wait_interval = 10
elapsed = 0

while elapsed < max_wait:
    resp = batch.describe_compute_environments(computeEnvironments=[CE_NAME])
    if resp['computeEnvironments']:
        status = resp['computeEnvironments'][0]['status']
        if status == 'VALID':
            print(f"  ✓ Compute environment is VALID")
            break
        elif status == 'INVALID':
            reason = resp['computeEnvironments'][0].get('statusReason', 'Unknown')
            print(f"  ✗ Compute environment is INVALID: {reason}")
            raise Exception(f"Compute environment failed: {reason}")
        else:
            print(f"  Status: {status}, waiting... ({elapsed}s elapsed)")
    time.sleep(wait_interval)
    elapsed += wait_interval

if elapsed >= max_wait:
    print(f"  ⚠️  Timeout waiting for compute environment (status may still be creating)")
    print(f"     Continuing anyway - queue creation may fail")

# Check if job queue already exists
print(f"Checking if job queue '{QUEUE_NAME}' exists...")
try:
    resp = batch.describe_job_queues(jobQueues=[QUEUE_NAME])
    if resp['jobQueues']:
        print(f"  ✓ Job queue '{QUEUE_NAME}' already exists")
        queue_exists = True
    else:
        queue_exists = False
except Exception as e:
    queue_exists = False

if not queue_exists:
    print(f"Creating job queue '{QUEUE_NAME}' ...")
    resp_queue = batch.create_job_queue(
        jobQueueName=QUEUE_NAME,
        state="ENABLED",
        priority=int(os.getenv("BATCH_QUEUE_PRIORITY", "1")),
        computeEnvironmentOrder=[
            {"order": 1, "computeEnvironment": CE_NAME}
        ]
    )
    print("Job queue created:", resp_queue)

# Check if job definition already exists and is ACTIVE
print(f"Checking if job definition '{JOB_DEF_NAME}' exists...")
try:
    resp = batch.describe_job_definitions(
        jobDefinitionName=JOB_DEF_NAME,
        status='ACTIVE'
    )
    if resp['jobDefinitions']:
        print(f"  ✓ Job definition '{JOB_DEF_NAME}' already exists")
        jobdef_exists = True
    else:
        jobdef_exists = False
except Exception as e:
    jobdef_exists = False

if not jobdef_exists:
    print(f"Registering job definition '{JOB_DEF_NAME}' ...")

    # Task role for container AWS API access (S3, etc.)
    task_role_arn = f"arn:aws:iam::{account_id}:role/AlgoTuneBatchTaskRole"

    resp_jobdef = batch.register_job_definition(
        jobDefinitionName=JOB_DEF_NAME,
        type="container",
        containerProperties={
            "image": DOCKER_IMAGE,
            "executionRoleArn": ecs_task_exec_role,
            "jobRoleArn": task_role_arn,  # Container's AWS API access role
            "vcpus": int(os.getenv("BATCH_JOB_VCPUS", "1")),
            "memory": int(os.getenv("BATCH_JOB_MEMORY_MB", "16000")),
            "command": []  # we'll override at submission
            # Note: Using default CloudWatch logging + S3 uploads every 60s
        }
    )
    print("Job definition registered:", resp_jobdef)

print("🎉 AWS Batch compute environment, job queue, and job definition are ready!")
