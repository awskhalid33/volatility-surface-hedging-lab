from vol_surface_hedging_lab.visualisation import _histogram, generate_visual_artifacts


def _example_payloads() -> tuple[dict, dict, dict, dict]:
    surface_result = {
        "per_maturity": [
            {
                "maturity": 0.5,
                "forward": 100.0,
                "rows": [
                    {"strike": 90.0, "implied_vol": 0.24},
                    {"strike": 100.0, "implied_vol": 0.20},
                    {"strike": 110.0, "implied_vol": 0.23},
                ],
                "svi_fit": {
                    "params": {
                        "a": 0.02,
                        "b": 0.10,
                        "rho": -0.25,
                        "m": 0.0,
                        "sigma": 0.18,
                    }
                },
            }
        ]
    }
    model_result = {
        "summary": {
            "avg_in_sample_rmse_iv": {"svi": 0.002, "sabr": 0.004},
            "avg_oos_cv_rmse_iv": {"svi": 0.003, "sabr": 0.006},
        },
        "per_maturity": [
            {
                "maturity": 0.5,
                "sabr": {
                    "params": {"alpha": 0.2, "beta": 1.0, "rho": -0.1, "nu": 0.7}
                },
            }
        ],
    }
    hedging_result = {
        "terminal_pnl_by_strategy": {
            "unhedged": [-5.0, -2.0, 0.5, 2.0],
            "delta": [-1.2, -0.4, 0.2, 0.8],
            "delta-vega": [-0.9, -0.2, 0.1, 0.6],
        }
    }
    rolling_result = {
        "calibration_quality": {
            "per_date_rmse": {
                "2025-01-02": 0.0014,
                "2025-01-03": 0.0013,
            }
        }
    }
    return surface_result, model_result, hedging_result, rolling_result


def test_generate_visual_artifacts_creates_expected_files(tmp_path):
    surface, model, hedging, rolling = _example_payloads()
    generated = generate_visual_artifacts(
        surface_result=surface,
        model_result=model,
        hedging_result=hedging,
        rolling_result=rolling,
        output_dir=tmp_path,
    )

    assert len(generated) == 6
    assert (tmp_path / "fig_smile_fit.svg").exists()
    assert (tmp_path / "fig_model_rmse.svg").exists()
    assert (tmp_path / "fig_terminal_pnl_hist_unhedged.svg").exists()
    assert (tmp_path / "fig_terminal_pnl_hist_delta.svg").exists()
    assert (tmp_path / "fig_terminal_pnl_hist_delta-vega.svg").exists()
    assert (tmp_path / "fig_rolling_rmse_trend.svg").exists()
    assert (tmp_path / "fig_smile_fit.svg").read_text(encoding="utf-8").startswith("<svg")


def test_generate_visual_artifacts_supports_file_prefix(tmp_path):
    surface, model, hedging, _ = _example_payloads()
    generated = generate_visual_artifacts(
        surface_result=surface,
        model_result=model,
        hedging_result=hedging,
        rolling_result={"calibration_quality": {"per_date_rmse": {}}},
        output_dir=tmp_path,
        file_prefix="live_",
    )

    assert len(generated) == 5
    assert (tmp_path / "live_fig_smile_fit.svg").exists()
    assert (tmp_path / "live_fig_model_rmse.svg").exists()
    assert (tmp_path / "live_fig_terminal_pnl_hist_unhedged.svg").exists()
    assert (tmp_path / "live_fig_terminal_pnl_hist_delta.svg").exists()
    assert (tmp_path / "live_fig_terminal_pnl_hist_delta-vega.svg").exists()


def test_histogram_writes_placeholder_for_empty_values(tmp_path):
    path = tmp_path / "empty_hist.svg"
    _histogram(path=path, title="Empty", values=[])
    text = path.read_text(encoding="utf-8")
    assert "No values available" in text
