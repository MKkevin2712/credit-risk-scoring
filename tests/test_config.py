from credit_risk_scoring.config import DATA_RAW, ROOT, TABLES


def test_project_root_contains_pyproject():
    assert (ROOT / "pyproject.toml").exists()


def test_raw_data_dir_is_inside_project():
    assert DATA_RAW.is_relative_to(ROOT)


def test_main_tables_are_listed():
    assert {"application_train", "bureau", "previous_application"} <= set(TABLES)
