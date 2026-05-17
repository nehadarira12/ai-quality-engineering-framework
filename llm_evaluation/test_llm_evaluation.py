# llm_evaluation/test_llm_evaluation.py
# ─────────────────────────────────────────────────────────────
# Tests that validate the quality of AI-generated responses.
# Covers two critical failure modes in production AI systems:
#   1. Irrelevant responses — AI answers a different question
#   2. Hallucination — AI invents facts not in the context
#
# Built from real experience testing a patient-facing
# Voice AI platform in a US healthcare environment.
# ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from conftest import groq_model


class TestAnswerRelevancy:
    """
    Tests that AI responses are relevant to the question asked.
    An irrelevant response in a healthcare or fintech context
    is not just a UX issue — it erodes patient and user trust.
    """

    def test_prescription_delivery_options(self):
        """
        Patient asks about delivery options.
        AI should respond with relevant delivery information.
        """
        test_case = LLMTestCase(
            input="What are my options for receiving my prescription?",
            actual_output="You can choose home delivery or pick up from your nearest pharmacy location.",
            expected_output="Patient should be informed about available delivery and pickup options."
        )
        metric = AnswerRelevancyMetric(threshold=0.7, model=groq_model)
        assert_test(test_case, [metric])

    def test_appointment_scheduling_response(self):
        """
        Patient asks about booking an appointment.
        AI should respond with scheduling-related information.
        """
        test_case = LLMTestCase(
            input="How do I book an appointment with my doctor?",
            actual_output="You can book an appointment through the patient portal or by calling the clinic directly.",
            expected_output="Patient should be given clear instructions on how to schedule an appointment."
        )
        metric = AnswerRelevancyMetric(threshold=0.7, model=groq_model)
        assert_test(test_case, [metric])


class TestHallucinationDetection:
    """
    Tests that AI does not introduce facts not present in context.
    In healthcare and fintech, hallucinated facts can cause
    financial harm or incorrect medical decisions.
    """

    def test_copay_amount_not_invented(self):
        """
        Patient account has no copay information.
        AI should not invent a copay amount.
        This mirrors a real failure mode caught in
        Voice AI testing at Foundation Health.
        """
        test_case = LLMTestCase(
            input="What is my copay amount for this prescription?",
            actual_output="Your copay is $25 for this prescription.",
            expected_output="AI should state no copay information is available.",
            context=["Patient account shows no copay information available."]
        )
        metric = HallucinationMetric(threshold=0.5, model=groq_model)
        assert_test(test_case, [metric])

    def test_appointment_details_not_invented(self):
        """
        Patient record has no upcoming appointments.
        AI should not invent appointment details.
        """
        test_case = LLMTestCase(
            input="When is my next appointment?",
            actual_output="Your next appointment is on June 15th at 10am with Dr. Smith.",
            expected_output="AI should state no appointment information is available.",
            context=["Patient record shows no upcoming appointments scheduled."]
        )
        metric = HallucinationMetric(threshold=0.5, model=groq_model)
        assert_test(test_case, [metric])
