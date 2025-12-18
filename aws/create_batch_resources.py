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

# Docker image you built & pushed
ECR_IMAGE_URI = os.getenv("ECR_IMAGE_URI")

# Roles
ecs_instance_role = os.getenv("ECS_INSTANCE_ROLE", "ecsInstanceRole")
ecs_task_exec_role = os.getenv("ECS_TASK_EXEC_ROLE", "ecsTaskExecutionRole")
batch_service_role = os.getenv("AWS_BATCH_SERVICE_ROLE")  # will be auto-created once

# Network config
subnets = os.getenv("AWS_SUBNET_ID", "").split(",")
security_groups = os.getenv("AWS_SG_ID", "").split(",")

print(f"Creating compute environment '{CE_NAME}' ...")
resp_ce = batch.create_compute_environment(
    computeEnvironmentName=CE_NAME,
    type="MANAGED",
    state="ENABLED",
    computeResources={
        "type": "EC2",
        "minvCpus": 0,
        "maxvCpus": int(os.getenv("BATCH_MAX_VCPUS", "64")),
        "desiredvCpus": 0,
        "instanceTypes": os.getenv("BATCH_INSTANCE_TYPES", "c6a.4xlarge").split(","),
        "subnets": subnets,
        "securityGroupIds": security_groups,
        "instanceRole": ecs_instance_role,
    },
    serviceRole=batch_service_role  # AWSServiceRoleForBatch
)

print("Compute Environment creation request submitted:", resp_ce)

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

print(f"Registering job definition '{JOB_DEF_NAME}' ...")
resp_jobdef = batch.register_job_definition(
    jobDefinitionName=JOB_DEF_NAME,
    type="container",
    containerProperties={
        "image": ECR_IMAGE_URI,
        "executionRoleArn": ecs_task_exec_role,
        "vcpus": int(os.getenv("BATCH_JOB_VCPUS", "1")),
        "memory": int(os.getenv("BATCH_JOB_MEMORY_MB", "16000")),
        "command": [],  # we'll override at submission
        # Disable CloudWatch logging - we'll capture logs to files instead
        "logConfiguration": {
            "logDriver": "none"
        }
    }
)
print("Job definition registered:", resp_jobdef)

print("🎉 AWS Batch compute environment, job queue, and job definition are ready!")
