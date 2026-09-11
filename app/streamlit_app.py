"""Simple Streamlit demo for Multimodal Materials Property Predictor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from materials_ai.data.dataset import load_processed_frame
from materials_ai.utils import project_root

st.set_page_config(
    page_title="Materials Band-Gap Predictor",
    page_icon="⬡",
    layout="wide",
)

ROOT = project_root()


@st.cache_resource
def load_predictor():
    from materials_ai.inference.predictor import MaterialsPredictor

    ckpt = ROOT / "outputs" / "models" / "multimodal_best.pt"
    processed = ROOT / "data" / "processed"
    if not ckpt.exists():
        return None, f"Checkpoint not found at {ckpt}. Train the model first."
    return MaterialsPredictor(checkpoint_path=ckpt, processed_dir=processed), None


@st.cache_data
def load_catalog() -> pd.DataFrame | None:
    processed = ROOT / "data" / "processed"
    try:
        return load_processed_frame(processed)
    except FileNotFoundError:
        return None


def main() -> None:
    st.title("Multimodal Materials Property Predictor")
    st.caption("Band-gap regression from structure images + tabular descriptors")

    catalog = load_catalog()
    predictor, err = load_predictor()
    if err:
        st.warning(err)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input")
        mode = st.radio("Select input mode", ["Catalog material", "Upload structure (CIF)"])
        material_id = None
        upload = None
        if mode == "Catalog material":
            if catalog is None:
                st.error("Processed dataset missing. Run scripts/prepare_dataset.py")
                return
            options = catalog["material_id"].tolist()
            material_id = st.selectbox("Material ID", options)
            row = catalog.loc[catalog["material_id"] == material_id].iloc[0]
            st.write(f"**Formula:** {row['formula']}")
            st.write(f"**Crystal system:** {row.get('crystal_system', 'n/a')}")
            if Path(row["image_path"]).exists():
                st.image(str(row["image_path"]), caption="Structure visualization", width=280)
            with st.expander("Extracted descriptors"):
                from materials_ai.data.descriptors import feature_names

                desc = {c: float(row[c]) for c in feature_names() if c in row}
                st.dataframe(pd.DataFrame(desc.items(), columns=["feature", "value"]))
        else:
            upload = st.file_uploader("Upload CIF / POSCAR", type=["cif", "vasp", "poscar"])

    with right:
        st.subheader("Prediction")
        if st.button("Run prediction", type="primary", disabled=predictor is None):
            assert predictor is not None
            with st.spinner("Running multimodal inference + MC dropout..."):
                if mode == "Catalog material":
                    result = predictor.predict(material_id=material_id, n_uncertainty=15)
                else:
                    if upload is None:
                        st.error("Please upload a structure file")
                        return
                    tmp = ROOT / "outputs" / "uploads" / upload.name
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    tmp.write_bytes(upload.getvalue())
                    result = predictor.predict(structure_path=tmp, n_uncertainty=15)

            st.metric("Predicted band gap", f"{result.prediction:.2f} {result.unit}")
            if result.uncertainty is not None:
                st.write(f"Uncertainty (MC-dropout σ): ±{result.uncertainty:.2f} {result.unit}")
                st.caption(
                    "This is predictive std under MC dropout, not a calibrated scientific CI."
                )
            st.write(f"Model version: `{result.model_version}`")
            st.write("Inputs used:", ", ".join(result.inputs_used))

            if result.explanation and result.explanation.get("image_path"):
                img_path = Path(result.explanation["image_path"])
                if img_path.exists():
                    st.image(Image.open(img_path), caption="Generated structure image", width=280)

            expl = ROOT / "outputs" / "explanations"
            reports = sorted(expl.glob("*_report.txt")) if expl.exists() else []
            cams = sorted(expl.glob("*_gradcam.png")) if expl.exists() else []
            if reports:
                st.subheader("Explainability")
                st.text(reports[-1].read_text(encoding="utf-8"))
            if cams:
                st.image(str(cams[-1]), caption="Grad-CAM attention", width=360)

    st.divider()
    st.markdown(
        "Scientific note: predictive performance indicates **correlation** with band gap "
        "under the chosen dataset/split — not causation or a materials discovery claim."
    )


if __name__ == "__main__":
    main()
