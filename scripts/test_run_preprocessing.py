from utils.run_preprocessing import run_preprocessing

if __name__ == "__main__":
    output = run_preprocessing("data/metadata.csv")

    print("Selected features:", output.feat_labels[:5])
    print("Selected algorithms:", output.algo_labels[:5])
    print("Feature matrix shape:", output.x.shape)
    print("Algorithm matrix shape:", output.y.shape)
