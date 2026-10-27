# Capstone-CyberSec-ML-LLM

UTS 32040 Industry Project — Spring 2026

## Project

Behavioural anomaly detection for potential insider-threat activity in a synthetic banking employee environment.

The project investigates role-specific normal employee behaviour using synthetic banking audit logs, behavioural anomaly detection, and contextual analysis.

## Team

- Fachrul Reza
- Christopher Gustaviano

## Dashboard
https://capstone-cybersec.streamlit.app/

## Current Status

Project rebuild and experimental implementation in progress.

## Important Scope

This project uses synthetic banking employee audit logs.

An anomalous behaviour identified by the system represents unusual behaviour requiring further investigation. It does not by itself establish malicious intent or confirm an insider threat.


## Reproducibility and Experimental Evidence

The project uses synthetic banking audit logs.

### Replicated Evaluation Results

The `outputs/robustness/` directory contains:

- `replication_metrics.csv` — individual replication results.
- `replication_summary.csv` — aggregated metrics and 95% confidence intervals.
- `final_performance_table.csv` — formatted performance results used in the final report.
- `replication_case_audit.csv` — case-level experimental audit records.

These files support independent verification of the replicated evaluation results.

### Verification Example

For the Subtle condition using deterministic rules:

- 50 replications
- 300 employee-days per replication
- Mean accuracy: 84.97%
- 95% confidence interval: 84.30–85.63%

Full experiment reproduction requires the corresponding code, configurations, and execution environment.
