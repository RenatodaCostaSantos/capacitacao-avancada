# ==========================================
# TELECOM ANOMALY — TREINO (entrypoint)
# Execute: python train.py
# ==========================================

from core.pipeline import (
    PipelineConfig, load_files,
    clean_dataframe, create_telecom_features, train
)

FILES = {
    "normal": [
        r"data\carbonZ_2018-07-18-16-37-39_1_no_failure-diagnostics.csv",
        r"data\carbonZ_2018-07-30-16-39-00_3_no_failure-diagnostics.csv",
        r"data\carbonZ_2018-09-11-14-16-55_no_failure-diagnostics.csv",
        r"data\carbonZ_2018-09-11-14-41-38_no_failure-diagnostics.csv",
        r"data\carbonZ_2018-10-05-14-34-20_1_no_failure-diagnostics.csv",
    ],
    "failure": [
        r"data\carbonZ_2018-07-18-15-53-31_2_engine_failure-diagnostics.csv",
        r"data\carbonZ_2018-07-30-16-39-00_1_engine_failure-diagnostics.csv",
        r"data\carbonZ_2018-07-30-16-39-00_2_engine_failure-diagnostics.csv",
        r"data\carbonZ_2018-09-11-11-56-30_engine_failure-diagnostics.csv",
        #r"data\carbonZ_2018-09-11-14-22-07_1_engine_failure-diagnostics.csv",
    ],
}

if __name__ == "__main__":
    config = PipelineConfig(
        contamination=0.3,
        n_estimators=800,
        random_state=42,
        threshold_percentile=50.0,
    )

    # 1. carregar
    df_normal_raw, df_failure_raw = load_files(FILES)
    print("\n✅ Dados carregados com sucesso!")
    print(df_normal_raw.shape, df_failure_raw.shape)
    # 2. limpar
    df_normal_clean  = clean_dataframe(df_normal_raw)
    df_failure_clean = clean_dataframe(df_failure_raw)
    print("✅ Dados limpos e prontos para feature engineering!")
    print(df_normal_clean.shape, df_failure_clean.shape)

    # 3. features
    df_normal_feat  = create_telecom_features(df_normal_clean)
    print(df_normal_feat)
    df_failure_feat = create_telecom_features(df_failure_clean)
    print("✅ Features criadas com sucesso! Prontos para o treino!")
    print(df_normal_feat.shape, df_failure_feat.shape)

    # 4. treinar + salvar
    result = train(df_normal_feat, df_failure_feat, config)

    print("\n✅ Treino concluído!")
    print(f"   F1 Score : {result.metrics['f1']}")
    print(f"   Recall   : {result.metrics['recall']}")
    print(f"   Precision: {result.metrics['precision']}")
    print(f"   Threshold: {result.metrics['threshold']}")
    print(f"\n   Modelos salvos em: models/")


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

    
