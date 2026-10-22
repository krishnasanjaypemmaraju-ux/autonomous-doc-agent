from app.agents.amounts import describe_amounts, indian_grouping


def test_indian_grouping():
    assert indian_grouping(10_000_000) == "1,00,00,000"
    assert indian_grouping(500_000) == "5,00,000"
    assert indian_grouping(1_234) == "1,234"
    assert indian_grouping(999) == "999"


def test_ten_million_is_one_crore_not_ten():
    [reading] = describe_amounts("my buget is 10000000")
    assert reading["indian"] == "1,00,00,000 = 1 crore"
    assert reading["international"] == "10,000,000 = 10 million"


def test_lakh_and_existing_commas():
    [reading] = describe_amounts("budget of 5,00,000 for 4 members")
    assert reading["indian"] == "5,00,000 = 5 lakh"
    assert reading["international"] == "500,000 = 500 thousand"


def test_budget_parts_must_fit_the_total():
    from app.schemas import budget_overrun

    over = ["Total budget: INR 1,00,00,000 (1 crore)",
            "Estimated direct construction cost: INR 90,00,000",
            "Estimated soft costs (permits, design, fees): INR 5,00,000",
            "Estimated contingency: INR 9,50,000",
            "Contingency: 10 % of direct costs",
            "Project lead: Rahul Sharma"]
    message = budget_overrun(over)
    assert message and "1,04,50,000".replace(",", "") in message.replace(",", "")

    fits = ["Total budget: ₹5,00,000", "Accommodation budget: ₹2,00,000",
            "Transport budget: ₹1,20,000", "Contingency reserve: ₹30,000"]
    assert budget_overrun(fits) is None
    assert budget_overrun(["Total budget: $10M", "Build cost: $9.5M", "Reserve: $0.5M"]) is None
    assert budget_overrun(["Total budget: $10M", "Build cost: $9.5M", "Reserve: $1M"])


def test_cost_table_without_total_row_must_fit_budget():
    import pytest
    from pydantic import ValidationError

    from app.schemas import SectionContent, total_budget

    budget = total_budget(["Total budget: INR 1,00,00,000 (1 crore)", "Lead: Asha"])
    assert budget == 10_000_000
    table = {"type": "table", "headers": ["Item", "Estimated Cost (INR)", "Cost per sq m"],
             "rows": [["Structure", "45,00,000", "22,500"], ["Finishes", "40,00,000", "20,000"],
                      ["Services", "19,00,000", "9,500"]]}
    with pytest.raises(ValidationError, match="more than the total budget"):
        SectionContent.model_validate({"blocks": [table]}, context={"budget_total": budget})
    # Without a known budget (or within it) the same table is fine.
    SectionContent.model_validate({"blocks": [table]})
    SectionContent.model_validate({"blocks": [table]}, context={"budget_total": 2 * budget})


def test_placeholders_are_rejected():
    import pytest
    from pydantic import ValidationError

    from app.schemas import SectionContent, placeholders

    assert placeholders("Owner: [Requester Name], visiting City A with Vendor X") == \
        ["City A", "[Requester Name]"]
    assert placeholders("Plan for Rahul and Asha in Jaipur [see section 3]") == []
    with pytest.raises(ValidationError, match="placeholders"):
        SectionContent(blocks=[{"type": "paragraph", "text": "Signed by [Your Name]."}])


def test_years_small_numbers_and_duplicates_are_skipped():
    readings = describe_amounts("In 2027 we have 12 people, 25000 and 25000 again, 1500000")
    assert [r["as_written"] for r in readings] == ["25000", "1500000"]
    assert readings[1]["indian"] == "15,00,000 = 15 lakh"
    assert readings[1]["international"] == "1,500,000 = 1.5 million"
