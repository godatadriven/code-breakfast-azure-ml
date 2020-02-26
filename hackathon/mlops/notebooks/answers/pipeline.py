preprocess_step = PythonScriptStep(
    name="preprocess",
    script_name="scripts/preprocess.py",
    arguments=["--input_dir", train_data, "--output_dir", preprocessed_data],
    inputs=[train_data],
    outputs=[preprocessed_data],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory=str(model_dir),
    allow_reuse=False,
)

train_step = PythonScriptStep(
    name="train",
    script_name="scripts/train.py",
    arguments=["--input_dir", preprocessed_data, "--model_dir", model_data],
    inputs=[preprocessed_data],
    outputs=[model_data],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory=str(model_dir),
    allow_reuse=False,
)

evaluate_step = PythonScriptStep(
    name="evaluate",
    script_name="scripts/evaluate.py",
    arguments=["--input_dir", preprocessed_data, "--model_dir", model_data],
    inputs=[preprocessed_data, model_data],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory=str(model_dir),
    allow_reuse=False,
)

register_step = PythonScriptStep(
    name="register",
    script_name="scripts/register.py",
    arguments=["--model_dir", model_data, "--model_name", "titanic"],
    inputs=[model_data],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory=str(model_dir),
    allow_reuse=False,
)

register_step.run_after(evaluate_step)

pipeline = Pipeline(
    workspace=workspace,
    steps=[preprocess_step, train_step, evaluate_step, register_step],
)
pipeline.validate()