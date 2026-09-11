# strict_navigation

严格导航研究阶段，原目录名strict_nav_20260910。三种子与最终独立确认已完成；旧文件中待运行文字仅是阶段历史。

[返回上一级](../README.md)

## 内容

| 文件或目录 | 说明 |
|---|---|
| [final_confirmation_v1](final_confirmation_v1/README.md) | 进入下一层说明。 |
| [final_geometry_v1](final_geometry_v1/README.md) | 进入下一层说明。 |
| [maps](maps/README.md) | 进入下一层说明。 |
| [oracle_waypoint_formal_v1](oracle_waypoint_formal_v1/README.md) | 进入下一层说明。 |
| [oracle_waypoint_t16_v1](oracle_waypoint_t16_v1/README.md) | 进入下一层说明。 |
| [oracle_waypoint_v1](oracle_waypoint_v1/README.md) | 进入下一层说明。 |
| [retrieval_seed_repeat_v1](retrieval_seed_repeat_v1/README.md) | 进入下一层说明。 |
| [splits](splits/README.md) | 进入下一层说明。 |
| [topology_bridge_v1](topology_bridge_v1/README.md) | 进入下一层说明。 |
| [topology_dev_v2](topology_dev_v2/README.md) | 进入下一层说明。 |
| [topology_stress_v1](topology_stress_v1/README.md) | 进入下一层说明。 |
| [PLAN.md](PLAN.md) | 说明或阶段报告。 |
| [analyze_continuity.py](analyze_continuity.py) | Paired validation comparisons specifically for continuity and action bounds. |
| [audit_collection_smoke.py](audit_collection_smoke.py) | 核验入口，先查对应冻结协议。 |
| [audit_executed_trace.py](audit_executed_trace.py) | Audit real execution before first terminal; allow official post-terminal NaN padding. |
| [audit_final_confirmation.py](audit_final_confirmation.py) | Final frozen confirmation audit; no adaptive method selection or new training. |
| [audit_final_confirmation_noop_replay.py](audit_final_confirmation_noop_replay.py) | Audit-only amendment: replay zero-action boundary projection; frozen experiments unchanged. |
| [audit_final_confirmation_pre_argv_schema.py](audit_final_confirmation_pre_argv_schema.py) | Final frozen confirmation audit; no adaptive method selection or new training. |
| [audit_geometry_collections.py](audit_geometry_collections.py) | Audit both completed controlled collections before authorizing their training. |
| [audit_geometry_control_pair.py](audit_geometry_control_pair.py) | Audit the finite controlled-geometry pair and publish paired development results. |
| [audit_geometry_control_pair_pre_identitycheck.py](audit_geometry_control_pair_pre_identitycheck.py) | Audit the finite controlled-geometry pair and publish paired development results. |
| [audit_geometry_control_pair_pre_paddingfix.py](audit_geometry_control_pair_pre_paddingfix.py) | Audit the finite controlled-geometry pair and publish paired development results. |
| [audit_geometry_smokes.py](audit_geometry_smokes.py) | CPU audit of controlled-data model interface checks; not an accuracy evaluation. |
| [audit_oracle_router_smokes.py](audit_oracle_router_smokes.py) | Learned oracle-router smoke audit; checks passthrough against frozen unmodified baseline. |
| [audit_oracle_threshold_cpu.py](audit_oracle_threshold_cpu.py) | CPU closed-loop route switching at8/16px, unlike nominal exact-waypoint feasibility. |
| [audit_oracle_waypoint_cpu.py](audit_oracle_waypoint_cpu.py) | Finite CPU checks for oracle route feasibility and non-mutating goal-image substitution. |
| [audit_oracle_waypoint_formal.py](audit_oracle_waypoint_formal.py) | Frozen paired oracle waypoint diagnostic; no extra replans or changed terminal goals. |
| [audit_oracle_waypoint_t16.py](audit_oracle_waypoint_t16.py) | Reuse frozen full trajectory audit with explicit per-arm tolerance and add8vs16 comparisons. |
| [audit_retrieval_geometry_stage.py](audit_retrieval_geometry_stage.py) | Audit completed geometry retrieval outputs and record conditional paired evidence. |
| [audit_three_seed_baseline.py](audit_three_seed_baseline.py) | Full-budget three-seed baseline audit, with shared-case dependence explicit. |
| [audit_three_seed_baseline_pre_legacy_schema.py](audit_three_seed_baseline_pre_legacy_schema.py) | Full-budget three-seed baseline audit, with shared-case dependence explicit. |
| [audit_three_seed_controls.py](audit_three_seed_controls.py) | Three-seed development controls, audited separately from learned transfers. |
| [audit_three_seed_transfers.py](audit_three_seed_transfers.py) | Audit three-seed temporal and retrieval transfers before freezing final confirmation. |
| [audit_topology_adapter.py](audit_topology_adapter.py) | Finite CPU-only random/no-op checks for custom topology -> strict evaluation interface. |
| [audit_topology_environment.py](audit_topology_environment.py) | CPU-only finite topology feasibility audit. Never train or select a model here. |
| [audit_topology_model_bridge.py](audit_topology_model_bridge.py) | Verify every learned interface, then report matched-engine bridge without topology claims. |
| [audit_topology_model_bridge_pre_paddingfix.py](audit_topology_model_bridge_pre_paddingfix.py) | Verify every learned interface, then report matched-engine bridge without topology claims. |
| [audit_topology_stress.py](audit_topology_stress.py) | Audit frozen topology stress results, including baselines and conditional paired intervals. |
| [build_retrieval_seed.py](build_retrieval_seed.py) | Reuse the frozen retrieval construction with a seed-specific encoded training cache. |
| [collect_geometry_control.py](collect_geometry_control.py) | Matched collector controls: same frames, policy mixture and seeds; change training geometry only. |
| [describe_oracle_failures.py](describe_oracle_failures.py) | Describe the current target at failed oracle episodes without causal relabeling. |
| [diagnose_navigation.py](diagnose_navigation.py) | Shared-candidate intervention audit. Simulator state only used in oracle diagnostics. |
| [env.sh](env.sh) | 历史环境/运行包装器；先审查路径与资源。 |
| [eval_oracle_waypoints.py](eval_oracle_waypoints.py) | Inject privileged goal routing without altering environment goals or flushing action buffers. |
| [eval_oracle_waypoints_t16.py](eval_oracle_waypoints_t16.py) | Same frozen oracle evaluator; only waypoint switching tolerance changes8->16px. |
| [eval_retrieval.py](eval_retrieval.py) | Independent wrapper so baseline evaluation remains unchanged. |
| [eval_strict.py](eval_strict.py) | Strict navigation evaluation: frozen cases/scaling; measured compute, no protocol mixing. |
| [eval_topology.py](eval_topology.py) | Read-only adaptation of strict evaluator to the separately audited topology simulator. |
| [experiment_queue.py](experiment_queue.py) | Single serial job queue. Locks own folder; finite jobs; stops at review gate. |
| [finalize_head_reporting.py](finalize_head_reporting.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [fit_temporal_head.py](fit_temporal_head.py) | Strict TRM-inspired head and shuffled-label control, selected solely by heldout pair loss. |
| [fix_collection_loader_smoke.py](fix_collection_loader_smoke.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [geometry_control_protocol.json](geometry_control_protocol.json) | 数值、参数或审计记录。 |
| [geometry_result_report.py](geometry_result_report.py) | Paired geometry results; no independent-trial pooling across maps. |
| [head_result_report.py](head_result_report.py) | Render current temporal-head evidence from audited outputs. |
| [integrate_geometry_report.py](integrate_geometry_report.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [integrate_transfer_report.py](integrate_transfer_report.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [jobs.before_geometry_path_repair.json](jobs.before_geometry_path_repair.json) | 数值、参数或审计记录。 |
| [jobs.json](jobs.json) | 数值、参数或审计记录。 |
| [launch_geometry_collections.py](launch_geometry_collections.py) | Launch two finite CPU-only collectors after online latency baselines; no GPU processes. |
| [oracle_waypoint_router.py](oracle_waypoint_router.py) | Privileged geometry/pose waypoint diagnostic, never a deployable visual method. |
| [patch_head_report_integration.py](patch_head_report_integration.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [prepare_controlled_eval.py](prepare_controlled_eval.py) | Prepare evaluation statistics routing for controlled-data models; defaults unchanged. |
| [prepare_final_geometry.py](prepare_final_geometry.py) | Freeze four fresh geometry goals before final method evaluation; no model inference. |
| [prepare_maps.py](prepare_maps.py) | Freeze controlled TwoRoom geometry cases; verify dynamics and expert solvability. |
| [prepare_split.py](prepare_split.py) | Freeze trajectory-group split, train-only statistics and cases before outcomes. |
| [prepare_validation_controls.py](prepare_validation_controls.py) | Reserve untouched episodes for confirmatory evaluation, and queue finite validation controls. |
| [preserve_report_history.py](preserve_report_history.py) | Preserve appended reviewed findings when refreshing the live generated snapshot. |
| [queue_final_confirmation.py](queue_final_confirmation.py) | Freeze a final five-method, three-seed confirmation; no future adaptive sweep. |
| [queue_geometry_collection.py](queue_geometry_collection.py) | 历史队列生成器；勿自动执行，先查对应冻结协议。 |
| [queue_geometry_control_pair.py](queue_geometry_control_pair.py) | Finite matched single/multi-layout control, gated on successful interface audits. |
| [queue_geometry_smokes.py](queue_geometry_smokes.py) | Queue finite controlled-data smoke tests after the existing full collection audit. |
| [queue_head_replications.py](queue_head_replications.py) | Extend only validated temporal-head mechanism: independent models and validation geometries. |
| [queue_oracle_router_smokes.py](queue_oracle_router_smokes.py) | Queue only four oracle-router integration smokes and audit, no full diagnostic yet. |
| [queue_oracle_t16.py](queue_oracle_t16.py) | One finite tolerance sensitivity, after existing cross-seed interventions. |
| [queue_oracle_waypoint_formal.py](queue_oracle_waypoint_formal.py) | Queue a finite oracle-route experiment after third-seed baseline evaluation. |
| [queue_retrieval_controls.py](queue_retrieval_controls.py) | Finite validation-only retrieval controls, after initial map baseline. |
| [queue_retrieval_maps.py](queue_retrieval_maps.py) | Test whether dynamic prediction adds value to retrieval on four validation geometries. |
| [queue_retrieval_seed_repeats.py](queue_retrieval_seed_repeats.py) | Complete a finite three-encoder-seed key retrieval comparison on development goals. |
| [queue_seed_replications.py](queue_seed_replications.py) | Append two independent training seeds and fixed baseline replications; no sweep. |
| [queue_temporal_controls.py](queue_temporal_controls.py) | Queue validated lightweight temporal-head adaptation and independent-seed planning controls. |
| [queue_topology_bridge.py](queue_topology_bridge.py) | Finite learned-interface smoke and matched-simulator bridge; no full topology evaluation. |
| [queue_topology_stress.py](queue_topology_stress.py) | Queue finite three-topology stress tests after audited matched-engine bridge. |
| [reachability_model.py](reachability_model.py) | TRM-inspired temporal pair head. Added module; official LeWM remains unchanged. |
| [record_geometry_stage.py](record_geometry_stage.py) | Record repaired interface, finite next stage, and a sourced PLDM correction. |
| [repair_geometry_paths.py](repair_geometry_paths.py) | Repair controlled dataset paths; retain original failed job and artifacts. |
| [retrieval_prior.py](retrieval_prior.py) | Training-only visual trajectory retrieval. No test/map coordinates used. |
| [run_final_frozen.py](run_final_frozen.py) | Execute only a frozen final-confirmation job after checking its immutable inputs. |
| [smoke_retrieval.py](smoke_retrieval.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [status.json](status.json) | 数值、参数或审计记录。 |
| [summarize_candidates.py](summarize_candidates.py) | Summarize candidate interventions; no method fitting or outcome filtering. |
| [summarize_final_supplements.py](summarize_final_supplements.py) | Final descriptive supplements, no new evaluations or parameter selection. |
| [summarize_navigation.py](summarize_navigation.py) | Audit completed navigation outputs and update live research reports. No new experiments. |
| [summarize_navigation_pre_history_preserve.py](summarize_navigation_pre_history_preserve.py) | Audit completed navigation outputs and update live research reports. No new experiments. |
| [topology_env.py](topology_env.py) | Small topology stress environment. New conservative collision semantics are explicit. |
| [topology_env_pre_floatguard.py](topology_env_pre_floatguard.py) | Small topology stress environment. New conservative collision semantics are explicit. |
| [topology_env_pre_keywordfix.py](topology_env_pre_keywordfix.py) | Small topology stress environment. New conservative collision semantics are explicit. |
| [train_strict.py](train_strict.py) | Official LeWM objective; group-disjoint episode split and train-only scaling. |
| [trajectory_diagnostics.py](trajectory_diagnostics.py) | Geometry-assisted diagnosis of completed logged trajectories; never used by controllers. |
| [transfer_result_report.py](transfer_result_report.py) | Paired evidence for transfer mechanisms, separate source of prior and WM gains. |
| [validation_controls_20260910.json](validation_controls_20260910.json) | 数值、参数或审计记录。 |
