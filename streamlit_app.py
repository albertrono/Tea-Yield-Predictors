
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _local_css(css: str):
	st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


APP_CSS = '''
body {
  background: linear-gradient(180deg, #f0f7ef 0%, #ffffff 100%);
}
.card {
  background: white;
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(14, 30, 37, 0.08);
}
.brand {
  font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.kpi {
  font-size: 28px;
  font-weight: 700;
  color: #0b6b3a;
}
.muted { color: #6b7280; }
'''


def load_model(path: str):
	p = Path(path)
	if not p.exists():
		return None
	try:
		return joblib.load(p)
	except Exception:
		return None


def main():
	st.set_page_config(page_title="Agritea — Tea Yield Predictor", layout="wide", page_icon="🍃")
	_local_css(APP_CSS)

	# Header
	col1, col2 = st.columns([3, 1])
	with col1:
		st.markdown("<div class='brand'><h1 style='margin:0'>Agritea</h1><p class='muted'>Beautifully predict tea yield with ML</p></div>", unsafe_allow_html=True)
	with col2:
		st.image("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=400&q=60", width=120)

	st.markdown("---")

	model = load_model("best_xgboost_model.joblib")

	left, right = st.columns((1, 2))

	with left:
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.subheader("Inputs")
		input_mode = st.radio("Mode", ["Single sample", "Batch CSV"], index=0)
		st.write("")

		user_df = None

		if input_mode == "Single sample":
			if model is not None and hasattr(model, "feature_names_in_"):
				features = list(model.feature_names_in_)
				values = {}
				with st.form("single_form"):
					for f in features:
						values[f] = st.number_input(f, value=0.0, format="%.3f")
					st.form_submit_button("Predict")
				user_df = pd.DataFrame([values])
			else:
				st.info("Model does not expose feature names. You can upload a single-row CSV instead.")
				uploaded = st.file_uploader("Upload single-row CSV (headers = feature names)", type=["csv"])
				if uploaded is not None:
					user_df = pd.read_csv(uploaded)

		else:
			uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
			if uploaded is not None:
				user_df = pd.read_csv(uploaded)

		st.markdown("</div>", unsafe_allow_html=True)

		st.markdown("\n")
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.subheader("Model")
		if model is None:
			st.error("Could not find or load `best_xgboost_model.joblib` in repository root.")
			st.caption("Place the trained model file named best_xgboost_model.joblib in the project root.")
		else:
			st.success("Model loaded")
			if hasattr(model, "feature_importances_"):
				st.write("Feature importances available")
		st.markdown("</div>", unsafe_allow_html=True)

	with right:
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.subheader("Preview & Predictions")

		if user_df is None:
			st.info("Provide inputs (single row or upload CSV) to see predictions and charts.")
		else:
			st.markdown("**Preview**")
			st.dataframe(user_df.head())

			if model is None:
				st.warning("No model available — preview only.")
			else:
				try:
					preds = model.predict(user_df)
					if preds.ndim > 1 and preds.shape[1] == 1:
						preds = preds.ravel()
					# Normalize predictions to a 1-D array and present as weight in kgs
					out_preds = np.asarray(preds).ravel()
					weight_kg = np.round(out_preds, 3)
					pred_df = pd.DataFrame({"weight_kg": weight_kg})

					st.markdown("**Predictions (weight in kgs)**")
					st.dataframe(pred_df.head())

					# Histogram (weight in kgs)
					fig = px.histogram(pred_df, x="weight_kg", nbins=25, title="Prediction distribution (kg)")
					st.plotly_chart(fig, use_container_width=True)

					csv = pred_df.to_csv(index=False).encode("utf-8")
					st.download_button("Download predictions as CSV", data=csv, file_name="agritea_predictions.csv", mime="text/csv")

					# Show feature importances if available
					if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
						fi = pd.DataFrame({"feature": model.feature_names_in_, "importance": model.feature_importances_})
						fi = fi.sort_values("importance", ascending=False).head(20)
						fig2 = px.bar(fi, x="importance", y="feature", orientation="h", title="Top feature importances")
						st.plotly_chart(fig2, use_container_width=True)

				except Exception as e:
					st.error(f"Prediction failed: {e}")

		st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("---")
	st.caption("Agritea — created with ❤️. Try uploading a CSV of soil/climate features or use single-sample mode.")


if __name__ == "__main__":
	main()

