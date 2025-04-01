if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorboard.backend.event_processing import event_accumulator

    # Top-level directory containing structured folders (e.g., "dataset-windowsize-gridpoints-sampletech-length")
    base_logdir = "logs"
    # Global output directory for the fused plots, organized by fused tag
    global_output_dir = "fused_plots"
    os.makedirs(global_output_dir, exist_ok=True)

    # Define scalar tags (exact matches)
    scalar_tags = [
        "train/actor_loss",
        "train/critic_loss",
        "time/fps"
    ]

    # For tensor logs, we want any tag that starts with these prefixes
    tensor_prefixes = [
        "action/Action",
        "action/Episode_Reward"
    ]


    # Helper function: returns the fused tag (the part after the slash)
    def fuse_tag(tag):
        parts = tag.split("/")
        if len(parts) > 1:
            return parts[1]  # e.g., "actor_loss" for "train/actor_loss" or "ActionB" for "action/ActionB"
        return tag


    # Define sliding window size for the moving average
    window_size = 50

    # Iterate over each structured folder in base_logdir
    for structured_folder in os.listdir(base_logdir):
        structured_folder_path = os.path.join(base_logdir, structured_folder)
        if not os.path.isdir(structured_folder_path):
            continue

        # Attempt to split the folder name into its components:
        parts = structured_folder.split("-")
        if len(parts) == 5:
            dataset, windowsize, gridpoints, sampletech, length = parts
        else:
            dataset = structured_folder
            windowsize = gridpoints = sampletech = length = ""

        # Global dictionary to collect time series for each fused tag for this structured folder.
        # Key: fused tag; Value: list of time series (each is a list of (step, value) tuples)
        global_data = {}

        # Iterate over each run folder in the structured folder.
        for run_folder in os.listdir(structured_folder_path):
            run_path = os.path.join(structured_folder_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            run_data = {}
            # List all event files in the run folder (assuming they contain "events.out.tfevents")
            event_files = [f for f in os.listdir(run_path) if "events.out.tfevents" in f]
            for event_file in event_files:
                full_event_path = os.path.join(run_path, event_file)
                guidance = {"scalars": 1000, "tensors": 1000}
                ea = event_accumulator.EventAccumulator(full_event_path, size_guidance=guidance)
                ea.Reload()

                # Process scalar tags
                available_scalar_tags = ea.Tags().get("scalars", [])
                for tag in scalar_tags:
                    if tag in available_scalar_tags:
                        fused = fuse_tag(tag)
                        if fused not in run_data:
                            run_data[fused] = []
                        events = ea.Scalars(tag)
                        for e in events:
                            run_data[fused].append((e.step, e.value))

                # Process tensor tags (for any tag starting with our desired prefixes)
                available_tensor_tags = ea.Tags().get("tensors", [])
                for avail_tag in available_tensor_tags:
                    for prefix in tensor_prefixes:
                        if avail_tag.startswith(prefix):
                            fused = fuse_tag(avail_tag)
                            if fused not in run_data:
                                run_data[fused] = []
                            events = ea.Tensors(avail_tag)
                            for e in events:
                                val = tf.make_ndarray(e.tensor_proto)
                                value = val.item() if val.size == 1 else val.mean()
                                run_data[fused].append((e.step, value))
                            break

            # Sort each run's data by step and add it to global_data.
            for tag, datapoints in run_data.items():
                if datapoints:
                    datapoints.sort(key=lambda x: x[0])
                    if tag not in global_data:
                        global_data[tag] = []
                    global_data[tag].append(datapoints)

        # For each fused tag from this structured folder, create a fused plot.
        for fused_tag, time_series_list in global_data.items():
            plt.figure()

            # Use a colormap to assign distinct colors for each run's series
            cmap = plt.get_cmap("tab10")
            n_series = len(time_series_list)
            colors = [cmap(i) for i in np.linspace(0, 1, n_series)]

            # First, plot the underlying time series with low opacity and thin lines (zorder=1)
            for idx, series in enumerate(time_series_list):
                steps = np.array([dp[0] for dp in series])
                values = np.array([dp[1] for dp in series])
                plt.plot(steps, values, color=colors[idx], alpha=0.3, linewidth=1, zorder=1)

            # Then, plot the sliding window mean for each series with a higher zorder so it is in front.
            for idx, series in enumerate(time_series_list):
                steps = np.array([dp[0] for dp in series])
                values = np.array([dp[1] for dp in series])
                if len(values) >= window_size:
                    window = np.ones(window_size) / window_size
                    smoothed = np.convolve(values, window, mode='valid')
                    smoothed_steps = steps[window_size - 1:]
                    # Plot the moving average with thinner line (linewidth=1.5) but full opacity,
                    # using zorder=10 to ensure it's in front.
                    plt.plot(smoothed_steps, smoothed, color=colors[idx], linestyle="--", linewidth=1.5, alpha=1.0,
                             zorder=10)

            # If the fused tag corresponds to an action metric, add a fixed horizontal line at 0.625 with high zorder.
            if fused_tag.lower().startswith("action"):
                plt.axhline(y=1.15, color="black", linestyle=":", linewidth=1.5, zorder=20)

            plt.xlabel("Step")
            plt.ylabel("Value")
            # Title in the format: "{fused_tag}: For {dataset} {windowsize} {gridpoints} {sampletech} {length}"
            title_str = f"{fused_tag}: For {dataset} {windowsize} {gridpoints} {sampletech} {length}"
            plt.title(title_str)
            plt.tight_layout()

            # Build filename in the desired format:
            plot_filename = f"{fused_tag}-{dataset}-{windowsize}-{gridpoints}-{sampletech}-{length}.png"
            # Create global folder for this fused tag
            tag_output_dir = os.path.join(global_output_dir, fused_tag)
            os.makedirs(tag_output_dir, exist_ok=True)
            plt.savefig(os.path.join(tag_output_dir, plot_filename))
            plt.close()

        print(f"Finished processing structured folder '{structured_folder}'")

    print("All fused plots saved in:", global_output_dir)
