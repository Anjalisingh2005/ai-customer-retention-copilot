"""ROI estimator math sanity checks."""
from __future__ import annotations

from src.models.roi_estimator import OFFER_CATALOG, estimate, rank_offers


def test_offer_keys_all_runnable():
    for key in OFFER_CATALOG:
        e = estimate(key, monthly_charge=75.0, baseline_churn_prob=0.6)
        assert e.offer_key == key
        assert e.expected_revenue_saved >= 0
        assert e.horizon_months == 12


def test_zero_baseline_means_zero_saved():
    e = estimate("loyalty_discount_15pct_12mo", monthly_charge=75.0, baseline_churn_prob=0.0)
    assert e.expected_revenue_saved == 0.0


def test_ranking_sorts_by_net_value():
    ranked = rank_offers(monthly_charge=80.0, baseline_churn_prob=0.55)
    nvs = [e.net_value for e in ranked]
    assert nvs == sorted(nvs, reverse=True)
