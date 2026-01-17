#!/usr/bin/env python3
"""
Fetch per-hour pricing for an EC2 instance type.
Uses AWS Pricing API for On-Demand and EC2 spot price history for Spot.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from dotenv import load_dotenv


REGION_LOCATION = {
    "af-south-1": "Africa (Cape Town)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-south-2": "Asia Pacific (Hyderabad)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ap-southeast-4": "Asia Pacific (Melbourne)",
    "ca-central-1": "Canada (Central)",
    "eu-central-1": "Europe (Frankfurt)",
    "eu-central-2": "Europe (Zurich)",
    "eu-north-1": "Europe (Stockholm)",
    "eu-south-1": "Europe (Milan)",
    "eu-south-2": "Europe (Spain)",
    "eu-west-1": "Europe (Ireland)",
    "eu-west-2": "Europe (London)",
    "eu-west-3": "Europe (Paris)",
    "il-central-1": "Israel (Tel Aviv)",
    "me-central-1": "Middle East (UAE)",
    "me-south-1": "Middle East (Bahrain)",
    "sa-east-1": "South America (Sao Paulo)",
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
}


def load_env() -> None:
    root_dotenv = Path(__file__).resolve().parents[1] / ".env"
    if root_dotenv.exists():
        load_dotenv(root_dotenv)

    aws_dotenv = Path(__file__).resolve().parent / ".env"
    if aws_dotenv.exists():
        load_dotenv(aws_dotenv)


def _extract_on_demand_price(price_list: list) -> Optional[float]:
    if not price_list:
        return None
    try:
        data = json.loads(price_list[0])
    except Exception:
        return None
    terms = data.get("terms", {}).get("OnDemand", {})
    for term in terms.values():
        for dimension in term.get("priceDimensions", {}).values():
            price = dimension.get("pricePerUnit", {}).get("USD")
            if price:
                try:
                    return float(price)
                except ValueError:
                    return None
    return None


def _get_products(pricing_client, filters: list[dict]) -> list:
    try:
        response = pricing_client.get_products(
            ServiceCode="AmazonEC2", Filters=filters, MaxResults=1
        )
    except Exception:
        return []
    return response.get("PriceList", [])


def get_on_demand_price(instance_type: str, region: str) -> Optional[float]:
    pricing = boto3.client("pricing", region_name="us-east-1")
    base_filters = [
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        {"Type": "TERM_MATCH", "Field": "productFamily", "Value": "Compute Instance"},
    ]

    location = REGION_LOCATION.get(region)
    if location:
        price_list = _get_products(
            pricing, base_filters + [{"Type": "TERM_MATCH", "Field": "location", "Value": location}]
        )
        price = _extract_on_demand_price(price_list)
        if price is not None:
            return price

    price_list = _get_products(
        pricing, base_filters + [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}]
    )
    return _extract_on_demand_price(price_list)


def get_spot_price(instance_type: str, region: str) -> Optional[float]:
    ec2 = boto3.client("ec2", region_name=region)
    try:
        response = ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(timezone.utc),
            MaxResults=20,
        )
    except Exception:
        return None

    history = response.get("SpotPriceHistory", [])
    if not history:
        return None

    latest_by_az: dict[str, dict] = {}
    for entry in history:
        az = entry.get("AvailabilityZone")
        ts = entry.get("Timestamp")
        if not az or not ts:
            continue
        current = latest_by_az.get(az)
        if current is None or ts > current.get("Timestamp"):
            latest_by_az[az] = entry

    prices = []
    for entry in latest_by_az.values():
        price_str = entry.get("SpotPrice")
        if not price_str:
            continue
        try:
            prices.append(float(price_str))
        except ValueError:
            continue
    if not prices:
        return None
    return sum(prices) / len(prices)


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(
        description="Fetch EC2 per-hour pricing for a given instance type."
    )
    parser.add_argument("--instance-type", required=True, help="EC2 instance type")
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: AWS_REGION or us-east-1)",
    )
    parser.add_argument(
        "--capacity-type",
        choices=["on-demand", "spot"],
        default="on-demand",
        help="Capacity type to price",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    if args.capacity_type == "spot":
        price = get_spot_price(args.instance_type, args.region)
        source = "ec2-spot-price-history"
    else:
        price = get_on_demand_price(args.instance_type, args.region)
        source = "aws-pricing"

    if price is None:
        print(
            f"WARNING: Could not fetch {args.capacity_type} price for "
            f"{args.instance_type} in {args.region}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        payload = {
            "capacity_type": args.capacity_type,
            "instance_type": args.instance_type,
            "region": args.region,
            "price_per_hour": round(price, 6),
            "currency": "USD",
            "source": source,
        }
        print(json.dumps(payload))
    else:
        print(f"{price:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
