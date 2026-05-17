# agent_output_validation/test_agent_outputs.py
# ─────────────────────────────────────────────────────────────
# Tests that validate outputs produced by AI agents.
# AI agents generate data catalogs, metric definitions,
# entity relationship diagrams, and business glossaries.
# These tests ensure agent outputs are complete, accurate,
# and safe before they are published to downstream systems.
# ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from conftest import groq_model


class TestCatalogEntryValidation:
    """
    Tests that AI-generated data catalog entries
    are complete and correctly structured.
    Incomplete catalog entries break downstream
    AI agents that depend on this metadata.
    """

    def test_catalog_entry_has_all_required_fields(self):
        """
        Every catalog entry must have table name,
        description, columns, owner, and sensitivity.
        """
        agent_output = {
            "table_name": "orders",
            "description": "Contains all customer purchase orders",
            "columns": ["order_id", "customer_id", "amount", "created_at"],
            "owner": "data_engineering",
            "sensitivity": "confidential"
        }

        required_fields = [
            "table_name",
            "description",
            "columns",
            "owner",
            "sensitivity"
        ]

        for field in required_fields:
            assert agent_output.get(field), \
                f"Catalog entry missing required field: '{field}'"

    def test_catalog_entry_columns_are_not_empty(self):
        """
        A table must have at least one column defined.
        An empty columns list means the agent failed
        to discover the table structure.
        """
        agent_output = {
            "table_name": "customers",
            "columns": ["customer_id", "name", "email", "created_at"]
        }

        assert len(agent_output["columns"]) > 0, \
            "Agent produced a catalog entry with no columns"

    def test_multiple_catalog_entries_are_unique(self):
        """
        Agent must not produce duplicate catalog entries
        for the same table.
        """
        catalog_entries = [
            {"table_name": "orders"},
            {"table_name": "customers"},
            {"table_name": "products"},
        ]

        table_names = [entry["table_name"] for entry in catalog_entries]
        assert len(table_names) == len(set(table_names)), \
            "Agent produced duplicate catalog entries"


class TestMetricsOutputValidation:
    """
    Tests that AI-generated metric definitions
    are complete and logically sound.
    """

    def test_metrics_output_has_required_fields(self):
        """
        Every metric definition must have a name,
        formula, description, and owner.
        """
        metrics_output = [
            {
                "name": "monthly_recurring_revenue",
                "formula": "SUM(subscription_amount) WHERE status = active",
                "description": "Total predictable monthly revenue from active subscriptions",
                "owner": "finance_team"
            },
            {
                "name": "customer_churn_rate",
                "formula": "lost_customers / total_customers_start * 100",
                "description": "Percentage of customers lost in a given period",
                "owner": "growth_team"
            }
        ]

        required_fields = ["name", "formula", "description", "owner"]

        for metric in metrics_output:
            for field in required_fields:
                assert metric.get(field), \
                    f"Metric '{metric.get('name', 'unknown')}' missing field: '{field}'"

    def test_metric_formula_uses_valid_operators(self):
        """
        Metric formulas must use valid mathematical
        or SQL operators. Invalid operators break
        downstream query execution.
        """
        formula = "SUM(subscription_amount) WHERE status = active"

        valid_operators = ["SUM", "COUNT", "AVG", "MAX", "MIN", "WHERE", "/", "*", "+", "-"]

        has_valid_operator = any(
            op in formula.upper()
            for op in valid_operators
        )

        assert has_valid_operator, \
            f"Metric formula '{formula}' contains no valid operators"


class TestAgentOutputAccuracy:
    """
    Tests that AI agent outputs are accurate
    and do not contain hallucinated information.
    """

    def test_agent_catalog_description_is_relevant(self):
        """
        The description an agent generates for a table
        must be relevant to the table name.
        """
        test_case = LLMTestCase(
            input="Generate a description for the table: orders",
            actual_output="The orders table contains records of all customer purchases including order id, amount, and timestamps.",
            expected_output="Description should be relevant to an orders or purchases table."
        )
        metric = AnswerRelevancyMetric(threshold=0.7, model=groq_model)
        assert_test(test_case, [metric])

    def test_agent_does_not_hallucinate_table_columns(self):
        """
        Agent must only reference columns that exist
        in the provided table schema.
        Hallucinated column names break SQL queries
        and downstream data pipelines.
        """
        test_case = LLMTestCase(
            input="What columns are in the orders table?",
            actual_output="The orders table contains order_id, customer_id, amount, and created_at.",
            expected_output="Should only mention columns present in the provided schema.",
            context=["Orders table schema: order_id, customer_id, amount, created_at"]
        )
        metric = HallucinationMetric(threshold=0.5, model=groq_model)
        assert_test(test_case, [metric])


class TestAgentGoalCompletion:
    """
    Tests that AI agents completed their assigned goals.
    An agent that partially completes a task produces
    incomplete data products that silently corrupt
    downstream analytics and reporting.
    """

    def test_catalog_generation_covered_all_tables(self):
        """
        When tasked to catalog all tables,
        the agent must produce an entry for every table.
        """
        available_tables = ["orders", "customers", "products", "payments"]

        agent_catalogued_tables = ["orders", "customers", "products", "payments"]

        for table in available_tables:
            assert table in agent_catalogued_tables, \
                f"Agent failed to catalog table: '{table}'"

    def test_metric_generation_covered_all_entities(self):
        """
        When tasked to generate metrics for all entities,
        the agent must produce metrics for every entity.
        """
        required_entities = ["Customer", "Order", "Product"]

        agent_generated_metrics = {
            "Customer": ["churn_rate", "lifetime_value"],
            "Order": ["average_order_value", "order_count"],
            "Product": ["revenue_per_product", "return_rate"]
        }

        for entity in required_entities:
            assert entity in agent_generated_metrics, \
                f"Agent did not generate metrics for entity: '{entity}'"
            assert len(agent_generated_metrics[entity]) > 0, \
                f"Agent generated empty metrics for entity: '{entity}'"
