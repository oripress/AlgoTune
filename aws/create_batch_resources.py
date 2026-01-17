#!/usr/bin/env python3

import os
import time
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

# Default resource names (legacy single-queue mode)
LEGACY_CE_NAME = os.getenv("BATCH_COMPUTE_ENV_NAME", "AlgoTuneCE")
LEGACY_QUEUE_NAME = os.getenv("BATCH_JOB_QUEUE_NAME", "AlgoTuneQueue")
JOB_DEF_NAME = os.getenv("BATCH_JOB_DEF_NAME", "AlgoTuneJobDef")

# Optional dual-queue mode (Spot + On-Demand)
SPOT_CE_NAME = os.getenv("BATCH_COMPUTE_ENV_NAME_SPOT")
SPOT_QUEUE_NAME = os.getenv("BATCH_JOB_QUEUE_NAME_SPOT")
ONDEMAND_CE_NAME = os.getenv("BATCH_COMPUTE_ENV_NAME_ONDEMAND")
ONDEMAND_QUEUE_NAME = os.getenv("BATCH_JOB_QUEUE_NAME_ONDEMAND")

USE_DUAL_QUEUES = all([SPOT_CE_NAME, SPOT_QUEUE_NAME, ONDEMAND_CE_NAME, ONDEMAND_QUEUE_NAME])

# Docker image you built & pushed (DockerHub)
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "algotune:latest")

# Optional: launch template for root volume sizing
root_volume_gb = os.getenv("BATCH_ROOT_VOLUME_GB")
launch_template_name = None

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
subnets = [s for s in os.getenv("AWS_SUBNET_ID", "").split(",") if s]
security_groups = [s for s in os.getenv("AWS_SG_ID", "").split(",") if s]


def get_env(name, fallback=None, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        if fallback:
            value = os.getenv(fallback, default)
        else:
            value = default
    return value


def get_env_int(name, fallback=None, default="0"):
    return int(get_env(name, fallback, default))


def get_env_list(name, fallback=None, default=""):
    value = get_env(name, fallback, default)
    return [item.strip() for item in value.split(",") if item.strip()]


# Ensure launch template exists if root volume override is requested.
def ensure_launch_template(template_name):
    if not root_volume_gb or not template_name:
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
            LaunchTemplateName=template_name,
            LaunchTemplateData=lt_data,
        )
        print(f"  ✓ Updated launch template '{template_name}' (root {volume_size}GB)")
    except ec2.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if (
            "InvalidLaunchTemplateName.NotFoundException" in error_code
            or "NotFoundException" in error_code
        ):
            # Template doesn't exist, create it
            try:
                ec2.create_launch_template(
                    LaunchTemplateName=template_name,
                    LaunchTemplateData=lt_data,
                    VersionDescription=f"root {volume_size}GB",
                )
                print(f"  ✓ Created launch template '{template_name}' (root {volume_size}GB)")
            except ec2.exceptions.ClientError as e2:
                error_code2 = e2.response.get("Error", {}).get("Code", "")
                if "UnauthorizedOperation" in error_code2:
                    print("  ⚠ Warning: No EC2 permissions to manage launch templates")
                    print(f"    Will attempt to use existing template '{template_name}'")
                else:
                    raise
        elif "UnauthorizedOperation" in error_code:
            print("  ⚠ Warning: No EC2 permissions to manage launch templates")
            print(f"    Will attempt to use existing template '{template_name}'")
        else:
            raise

    return template_name


if root_volume_gb:
    launch_template_name = get_env("BATCH_LAUNCH_TEMPLATE_NAME")
    if not launch_template_name:
        base_name = (
            get_env("BATCH_LAUNCH_TEMPLATE_BASE", None, "AlgoTuneBatch")
            if USE_DUAL_QUEUES
            else LEGACY_CE_NAME
        )
        launch_template_name = f"{base_name}-lt"
    ensure_launch_template(launch_template_name)


def build_compute_env_specs():
    if USE_DUAL_QUEUES:
        spot_min = get_env_int("BATCH_MIN_VCPUS_SPOT", "BATCH_MIN_VCPUS", "0")
        spot_max = get_env_int("BATCH_MAX_VCPUS_SPOT", "BATCH_MAX_VCPUS", "64")
        spot_desired = get_env_int(
            "BATCH_DESIRED_VCPUS_SPOT", "BATCH_DESIRED_VCPUS", str(spot_min)
        )
        ondemand_min = get_env_int("BATCH_MIN_VCPUS_ONDEMAND", "BATCH_MIN_VCPUS", "0")
        ondemand_max = get_env_int("BATCH_MAX_VCPUS_ONDEMAND", "BATCH_MAX_VCPUS", "64")
        ondemand_desired = get_env_int(
            "BATCH_DESIRED_VCPUS_ONDEMAND", "BATCH_DESIRED_VCPUS", str(ondemand_min)
        )

        spot_fleet_role = get_env(
            "BATCH_SPOT_IAM_FLEET_ROLE",
            None,
            f"arn:aws:iam::{account_id}:role/aws-ec2-spot-fleet-tagging-role",
        )

        return [
            {
                "name": SPOT_CE_NAME,
                "queue_name": SPOT_QUEUE_NAME,
                "resource_type": "SPOT",
                "min_vcpus": spot_min,
                "max_vcpus": spot_max,
                "desired_vcpus": spot_desired,
                "instance_types": get_env_list(
                    "BATCH_INSTANCE_TYPES_SPOT", "BATCH_INSTANCE_TYPES", "c6a.4xlarge"
                ),
                "queue_priority": get_env_int(
                    "BATCH_QUEUE_PRIORITY_SPOT", "BATCH_QUEUE_PRIORITY", "1"
                ),
                "spot_allocation_strategy": get_env(
                    "BATCH_SPOT_ALLOCATION_STRATEGY", None, "SPOT_CAPACITY_OPTIMIZED"
                ),
                "spot_bid_percentage": get_env_int("BATCH_SPOT_BID_PERCENTAGE", None, "100"),
                "spot_fleet_role": spot_fleet_role,
            },
            {
                "name": ONDEMAND_CE_NAME,
                "queue_name": ONDEMAND_QUEUE_NAME,
                "resource_type": "EC2",
                "min_vcpus": ondemand_min,
                "max_vcpus": ondemand_max,
                "desired_vcpus": ondemand_desired,
                "instance_types": get_env_list(
                    "BATCH_INSTANCE_TYPES_ONDEMAND", "BATCH_INSTANCE_TYPES", "c6a.4xlarge"
                ),
                "queue_priority": get_env_int(
                    "BATCH_QUEUE_PRIORITY_ONDEMAND", "BATCH_QUEUE_PRIORITY", "1"
                ),
            },
        ]

    legacy_min = get_env_int("BATCH_MIN_VCPUS", None, "0")
    legacy_max = get_env_int("BATCH_MAX_VCPUS", None, "64")
    legacy_desired = get_env_int("BATCH_DESIRED_VCPUS", None, str(legacy_min))

    return [
        {
            "name": LEGACY_CE_NAME,
            "queue_name": LEGACY_QUEUE_NAME,
            "resource_type": "EC2",
            "min_vcpus": legacy_min,
            "max_vcpus": legacy_max,
            "desired_vcpus": legacy_desired,
            "instance_types": get_env_list("BATCH_INSTANCE_TYPES", None, "c6a.4xlarge"),
            "queue_priority": get_env_int("BATCH_QUEUE_PRIORITY", None, "1"),
        }
    ]


def build_compute_resources(spec):
    resources = {
        "minvCpus": spec["min_vcpus"],
        "maxvCpus": spec["max_vcpus"],
        "desiredvCpus": spec["desired_vcpus"],
        "instanceTypes": spec["instance_types"],
        "subnets": subnets,
        "securityGroupIds": security_groups,
        "instanceRole": ecs_instance_role,
    }

    if spec["resource_type"] == "SPOT":
        resources["type"] = "SPOT"
        allocation = spec.get("spot_allocation_strategy")
        bid = spec.get("spot_bid_percentage")
        fleet_role = spec.get("spot_fleet_role")
        if allocation:
            resources["allocationStrategy"] = allocation
        if bid is not None:
            resources["bidPercentage"] = bid
        if fleet_role:
            resources["spotIamFleetRole"] = fleet_role
    else:
        resources["type"] = "EC2"

    if launch_template_name:
        resources["launchTemplate"] = {
            "launchTemplateName": launch_template_name,
            "version": "$Latest",
        }

    return resources


def ensure_compute_environment(spec):
    ce_name = spec["name"]

    # Check if compute environment already exists
    print(f"Checking if compute environment '{ce_name}' exists...")
    try:
        resp = batch.describe_compute_environments(computeEnvironments=[ce_name])
        if resp["computeEnvironments"]:
            print(f"  ✓ Compute environment '{ce_name}' already exists")
            ce_exists = True
            existing_ce = resp["computeEnvironments"][0]
        else:
            ce_exists = False
            existing_ce = None
    except Exception:
        ce_exists = False
        existing_ce = None

    if not ce_exists:
        print(f"Creating compute environment '{ce_name}' ...")

        ce_config = {
            "computeEnvironmentName": ce_name,
            "type": "MANAGED",
            "state": "ENABLED",
            "computeResources": build_compute_resources(spec),
        }

        # Only add serviceRole if it's not a service-linked role (service-linked roles are auto-used)
        if batch_service_role and "aws-service-role" not in batch_service_role:
            ce_config["serviceRole"] = batch_service_role

        resp_ce = batch.create_compute_environment(**ce_config)
        print("Compute Environment creation request submitted:", resp_ce)
        return

    current_state = existing_ce.get("state")
    current_resources = existing_ce.get("computeResources", {})
    current_type = current_resources.get("type")
    if current_type and current_type != spec["resource_type"]:
        print(
            f"  ⚠️  Warning: compute environment '{ce_name}' type is '{current_type}',"
            f" expected '{spec['resource_type']}'."
        )

    current_min = current_resources.get("minvCpus")
    current_max = current_resources.get("maxvCpus")
    current_desired = current_resources.get("desiredvCpus")
    current_lt = current_resources.get("launchTemplate", {})

    needs_update = False
    update_params = {"computeEnvironment": ce_name}

    # Re-enable if disabled (e.g., by Ctrl+C kill switch)
    if current_state == "DISABLED":
        print("  → Compute environment is DISABLED, re-enabling...")
        update_params["state"] = "ENABLED"
        needs_update = True

    # Ensure compute resources match current run settings (keeps ASG in sync).
    if (
        current_min != spec["min_vcpus"]
        or current_max != spec["max_vcpus"]
        or current_desired != spec["desired_vcpus"]
    ):
        print(
            "  → Updating compute resources to "
            f"min={spec['min_vcpus']}, max={spec['max_vcpus']}, desired={spec['desired_vcpus']}..."
        )
        update_params["computeResources"] = {
            "minvCpus": spec["min_vcpus"],
            "maxvCpus": spec["max_vcpus"],
            "desiredvCpus": spec["desired_vcpus"],
        }
        needs_update = True

    if spec["resource_type"] == "SPOT":
        spot_updates = {}
        allocation = spec.get("spot_allocation_strategy")
        bid = spec.get("spot_bid_percentage")
        fleet_role = spec.get("spot_fleet_role")
        if allocation and current_resources.get("allocationStrategy") != allocation:
            spot_updates["allocationStrategy"] = allocation
        if bid is not None and current_resources.get("bidPercentage") != bid:
            spot_updates["bidPercentage"] = bid
        if fleet_role and current_resources.get("spotIamFleetRole") != fleet_role:
            spot_updates["spotIamFleetRole"] = fleet_role
        if spot_updates:
            update_params.setdefault("computeResources", {}).update(spot_updates)
            needs_update = True

    if launch_template_name:
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


def wait_for_compute_environment(ce_name):
    print(f"Waiting for compute environment '{ce_name}' to become VALID...")
    max_wait = 300  # 5 minutes
    wait_interval = 10
    elapsed = 0

    while elapsed < max_wait:
        resp = batch.describe_compute_environments(computeEnvironments=[ce_name])
        if resp["computeEnvironments"]:
            status = resp["computeEnvironments"][0]["status"]
            if status == "VALID":
                print("  ✓ Compute environment is VALID")
                break
            if status == "INVALID":
                reason = resp["computeEnvironments"][0].get("statusReason", "Unknown")
                print(f"  ✗ Compute environment is INVALID: {reason}")
                raise Exception(f"Compute environment failed: {reason}")
            print(f"  Status: {status}, waiting... ({elapsed}s elapsed)")
        time.sleep(wait_interval)
        elapsed += wait_interval

    if elapsed >= max_wait:
        print("  ⚠️  Timeout waiting for compute environment (status may still be creating)")
        print("     Continuing anyway - queue creation may fail")


def ensure_job_queue(spec):
    queue_name = spec["queue_name"]
    ce_name = spec["name"]
    queue_priority = spec["queue_priority"]

    # Check if job queue already exists
    print(f"Checking if job queue '{queue_name}' exists...")
    try:
        resp = batch.describe_job_queues(jobQueues=[queue_name])
        if resp["jobQueues"]:
            print(f"  ✓ Job queue '{queue_name}' already exists")
            queue_exists = True
            existing_queue = resp["jobQueues"][0]
        else:
            queue_exists = False
            existing_queue = None
    except Exception:
        queue_exists = False
        existing_queue = None

    if not queue_exists:
        print(f"Creating job queue '{queue_name}' ...")
        resp_queue = batch.create_job_queue(
            jobQueueName=queue_name,
            state="ENABLED",
            priority=queue_priority,
            computeEnvironmentOrder=[{"order": 1, "computeEnvironment": ce_name}],
        )
        print("Job queue created:", resp_queue)
        return

    desired_order = [{"order": 1, "computeEnvironment": ce_name}]
    existing_order = existing_queue.get("computeEnvironmentOrder", []) if existing_queue else []
    existing_priority = existing_queue.get("priority") if existing_queue else None
    existing_state = existing_queue.get("state") if existing_queue else None

    if existing_order != desired_order or existing_priority != queue_priority or existing_state != "ENABLED":
        print(f"Updating job queue '{queue_name}' to use compute environment '{ce_name}' ...")
        resp_queue = batch.update_job_queue(
            jobQueue=queue_name,
            state="ENABLED",
            priority=queue_priority,
            computeEnvironmentOrder=desired_order,
        )
        print("Job queue updated:", resp_queue)
    else:
        print(f"  ✓ Job queue '{queue_name}' already configured for '{ce_name}'")


compute_env_specs = build_compute_env_specs()

for spec in compute_env_specs:
    ensure_compute_environment(spec)
    wait_for_compute_environment(spec["name"])
    ensure_job_queue(spec)

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

print("🎉 AWS Batch compute environments, job queues, and job definition are ready!")
