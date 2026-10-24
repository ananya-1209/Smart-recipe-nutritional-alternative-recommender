import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# PAGE CONFIGURATION — keeps the existing app's overall layout
# ============================================================
st.set_page_config(
    page_title="Smart Recipe & Nutritional Recommender",
    page_icon="🍲",
    layout="wide"
)

st.title("🍲 Smart Recipe & Nutritional Alternative Recommender")
st.markdown(
    """
    **AI-based recipe recommendation system using:**
    - TF-IDF ingredient similarity and cosine similarity
    - Nutritional similarity
    - Collaborative filtering from recipe ratings
    - Glycemic-index (GI)-aware ingredient substitutions
    - Nutritional comparisons and wellness information

    *Educational tool only, not medical advice. GI values and nutrition vary by
    variety, processing, ripeness, preparation, portion size, and the person.*
    """
)
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
RATINGS_FILE = os.path.join(BASE_DIR, "ratings.csv")
GI_FILE = os.path.join(BASE_DIR, "ingredient_gi.csv")

# ============================================================
# LOAD RECIPE DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    required = [
        "final_food_name", "Cleaned-Ingredients", "Calories (kcal)",
        "Carbohydrates (g)", "Protein (g)", "Fats (g)", "Free Sugar (g)",
        "Fibre (g)", "Sodium (mg)", "Calcium (mg)", "Iron (mg)",
        "Vitamin C (mg)", "Folate (µg)"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in data.csv: " + ", ".join(missing))

    df = df[required].copy().rename(columns={
        "final_food_name": "Recipe",
        "Cleaned-Ingredients": "Ingredients",
        "Calories (kcal)": "Calories",
        "Carbohydrates (g)": "Carbohydrates",
        "Protein (g)": "Protein",
        "Fats (g)": "Fats",
        "Free Sugar (g)": "Sugar",
        "Fibre (g)": "Fibre",
        "Sodium (mg)": "Sodium",
        "Calcium (mg)": "Calcium",
        "Iron (mg)": "Iron",
        "Vitamin C (mg)": "Vitamin_C",
        "Folate (µg)": "Folate"
    })
    df = df.dropna(subset=["Recipe", "Ingredients"]).copy()
    numeric = [
        "Calories", "Carbohydrates", "Protein", "Fats", "Sugar", "Fibre",
        "Sodium", "Calcium", "Iron", "Vitamin_C", "Folate"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(median) else median)
    df["Recipe"] = df["Recipe"].astype(str).str.strip()
    df["Ingredients"] = df["Ingredients"].astype(str)
    df = df[df["Recipe"] != ""].drop_duplicates(subset=["Recipe"])
    return df.reset_index(drop=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ data.csv was not found. Put data.csv in the same folder as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"❌ Could not load data.csv: {exc}")
    st.stop()

st.success(f"✅ Dataset loaded successfully! {len(df)} recipes available.")
m1, m2, m3 = st.columns(3)
m1.metric("Recipes", len(df))
m2.metric("Nutrition Features", 11)
m3.metric("Recommendation Method", "Hybrid")
st.divider()

# ============================================================
# GI REFERENCE DATA
# Approximate illustrative values; not a clinical GI database.
# ingredient_gi.csv, if supplied, overrides these entries.
# ============================================================
DEFAULT_GI = {
    "white rice": 73, "rice": 73, "brown rice": 50, "basmati rice": 58,
    "quinoa": 53, "oats": 55, "rolled oats": 55, "whole wheat": 55,
    "whole wheat flour": 55, "maida": 85, "white bread": 75, "bread": 70,
    "potato": 78, "sweet potato": 63, "corn": 52, "corn flour": 70,
    "sugar": 65, "jaggery": 65, "honey": 58, "banana": 51, "apple": 36,
    "mango": 51, "watermelon": 76, "chickpeas": 28, "chana": 28,
    "rajma": 24, "kidney beans": 24, "lentils": 32, "masoor dal": 32,
    "moong": 31, "dal": 32, "milk": 31, "yogurt": 35,
    "paneer": 30, "tofu": 15, "almond milk": 25, "cream": 35,
    "tomato": 15, "red bell pepper": 15, "cucumber": 15, "zucchini": 15,
    "carrot": 39, "peas": 48, "flour": 70, "semolina": 66,
    "rava": 66, "ghee": 0, "butter": 0, "olive oil": 0,
    "sesame oil": 0, "salt": 0, "spinach": 15, "broccoli": 15,
    "cauliflower": 15, "onion": 15, "garlic": 15, "turmeric": 15
}

# Base substitutions: (replacement, short explanation)
SUBSTITUTES = {
    "white rice": ("Quinoa", "Often provides more fibre and protein than white rice."),
    "rice": ("Quinoa or a portion of brown rice", "Compare the serving size and total carbohydrate amount."),
    "basmati rice": ("A smaller portion of brown rice or quinoa", "The portion size still matters for blood glucose."),
    "potato": ("Cauliflower or a modest portion of sweet potato", "Cauliflower is lower in carbohydrate; sweet potato is not automatically low-GI."),
    "sugar": ("Reduce the added sweetener; consider a suitable non-sugar option", "Jaggery and honey still contain sugars and are not automatically better for glucose."),
    "jaggery": ("Reduce or omit added jaggery", "Jaggery can raise blood glucose just like other added sugars."),
    "maida": ("Whole-wheat flour, or part whole-grain flour", "More fibre may help, but total carbohydrate and portion size remain important."),
    "white bread": ("Whole-grain bread", "Choose a higher-fibre option and check the label."),
    "cream": ("Plain unsweetened yogurt", "Can reduce saturated fat in some recipes; texture and cooking behaviour differ."),
    "paneer": ("Tofu", "A plant-based protein option; check the recipe's texture and nutrition."),
    "milk": ("Unsweetened almond or soy milk", "Choose an unsweetened product and compare protein and carbohydrate labels."),
    "butter": ("A measured amount of olive or sesame oil", "Provides unsaturated fats; calories are still substantial."),
    "ghee": ("A measured amount of olive or sesame oil", "Changes the fat profile, but does not make a recipe low-calorie."),
    "banana": ("Apple or berries, where suitable", "Fruit still contains carbohydrate; portion size matters."),
    "cream cheese": ("Plain unsweetened yogurt or cottage cheese", "Check consistency and nutrition for the specific product.")
}

def normalize_text(value):
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def ingredient_tokens(value):
    # Keep multiword ingredient phrases where possible, then individual tokens.
    return normalize_text(value).split()

@st.cache_data
def load_gi_table(path):
    gi = dict(DEFAULT_GI)
    if os.path.exists(path):
        try:
            gi_df = pd.read_csv(path)
            cols = {c.lower().strip(): c for c in gi_df.columns}
            ingredient_col = cols.get("ingredient") or cols.get("name")
            gi_col = cols.get("gi") or cols.get("glycemic index") or cols.get("glycemic_index")
            if ingredient_col and gi_col:
                for _, row in gi_df.iterrows():
                    name = normalize_text(row[ingredient_col])
                    value = pd.to_numeric(row[gi_col], errors="coerce")
                    if name and pd.notna(value):
                        gi[name] = float(np.clip(value, 0, 100))
        except Exception:
            pass
    return gi

gi_values = load_gi_table(GI_FILE)

def ingredient_gi(name):
    """Return a known approximate GI, or None when no reference value exists."""
    key = normalize_text(name)
    if key in gi_values:
        return float(gi_values[key])
    # Try matching a known phrase contained in the ingredient text.
    matches = [(k, v) for k, v in gi_values.items() if len(k) > 3 and k in key]
    if matches:
        return float(max(matches, key=lambda item: len(item[0]))[1])
    return None

def recipe_gi_score(ingredients):
    """
    Returns a relative lower-GI ingredient score (0..1) using only ingredients
    with a known reference value. It is NOT the recipe's measured GI.
    """
    names = ingredient_tokens(ingredients)
    found = []
    # Match longer GI names first to avoid 'rice' overriding 'brown rice'.
    for name in sorted(gi_values, key=len, reverse=True):
        if name in normalize_text(ingredients):
            found.append(gi_values[name])
    if not found:
        return 0.5, None, 0
    mean_gi = float(np.mean(found))
    return 1.0 - mean_gi / 100.0, mean_gi, len(found)

# ============================================================
# CONTENT + NUTRITION SIMILARITY
# ============================================================
@st.cache_data
def calculate_similarity(data):
    nutrition_cols = [
        "Calories", "Carbohydrates", "Protein", "Fats", "Sugar", "Fibre",
        "Sodium", "Calcium", "Iron", "Vitamin_C", "Folate"
    ]
    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
    tfidf = vectorizer.fit_transform(data["Ingredients"].fillna("").astype(str))
    ingredient_sim = cosine_similarity(tfidf)
    scaler = MinMaxScaler()
    nutrition_scaled = scaler.fit_transform(data[nutrition_cols])
    nutrition_sim = cosine_similarity(nutrition_scaled)
    hybrid = 0.7 * ingredient_sim + 0.3 * nutrition_sim
    return hybrid

with st.spinner("🔄 Preparing recommendation engine..."):
    hybrid_similarity = calculate_similarity(df)

# ============================================================
# OPTIONAL RATINGS DATA / COLLABORATIVE FILTERING
# ratings.csv schema: user_id, recipe, rating (1–5)
# ============================================================
def load_ratings(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["user_id", "recipe", "rating"])
    try:
        ratings = pd.read_csv(path)
        ratings.columns = [str(c).strip().lower() for c in ratings.columns]
        # Accept a few common column aliases.
        aliases = {}
        for col in ratings.columns:
            if col in ("user", "userid", "user id"):
                aliases[col] = "user_id"
            elif col in ("recipe_name", "item", "food", "food_name"):
                aliases[col] = "recipe"
            elif col in ("score", "stars"):
                aliases[col] = "rating"
        ratings = ratings.rename(columns=aliases)
        if not {"user_id", "recipe", "rating"}.issubset(ratings.columns):
            return pd.DataFrame(columns=["user_id", "recipe", "rating"])
        ratings = ratings[["user_id", "recipe", "rating"]].copy()
        ratings["recipe"] = ratings["recipe"].astype(str).str.strip()
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings = ratings.dropna(subset=["user_id", "recipe", "rating"])
        ratings = ratings[ratings["rating"].between(1, 5)]
        ratings = ratings[ratings["recipe"].isin(df["Recipe"])]
        return ratings
    except Exception:
        return pd.DataFrame(columns=["user_id", "recipe", "rating"])

ratings_df = load_ratings(RATINGS_FILE)

def collaborative_scores(recipe_names, ratings):
    """
    Item-based collaborative filtering using cosine similarity between
    recipe rating vectors. Returns normalized scores 0..1.
    """
    names = list(recipe_names)
    if ratings.empty or ratings["user_id"].nunique() < 2 or ratings["recipe"].nunique() < 2:
        return np.full(len(names), 0.5), False

    matrix = ratings.pivot_table(
        index="user_id", columns="recipe", values="rating", aggfunc="mean"
    )
    # Align rating columns to recipes in the current dataset.
    matrix = matrix.reindex(columns=names)
    item_user = matrix.fillna(0).T
    if item_user.shape[0] < 2:
        return np.full(len(names), 0.5), False
    sim = cosine_similarity(item_user)
    np.fill_diagonal(sim, 0)
    # Popularity/positive feedback weighted by ratings; neutral for unrated items.
    means = matrix.mean(axis=0).reindex(names)
    means = means.fillna(3.0)
    mean_norm = ((means - 1) / 4).to_numpy(dtype=float)
    if np.nanmax(sim) > 0:
        neighbour_score = sim @ mean_norm
        denom = sim.sum(axis=1)
        neighbour_score = np.divide(
            neighbour_score, denom, out=np.full_like(neighbour_score, 0.5), where=denom > 0
        )
        result = 0.65 * neighbour_score + 0.35 * mean_norm
    else:
        result = mean_norm
    result = np.nan_to_num(result, nan=0.5)
    return np.clip(result, 0, 1), True

cf_scores, cf_available = collaborative_scores(df["Recipe"], ratings_df)

# ============================================================
# TOP-LEVEL RECIPE SELECTOR AND NUTRITION DETAILS
# ============================================================
st.header("🍽️ Select a Recipe")
recipe_list = df["Recipe"].tolist()
selected_recipe_name = st.selectbox("Choose a recipe from the dataset:", recipe_list)
recipe_index = df.index[df["Recipe"] == selected_recipe_name][0]
selected_recipe = df.iloc[recipe_index]

st.divider()
st.header("📋 Selected Recipe")
st.subheader(selected_recipe["Recipe"])
st.write("**Ingredients:**", selected_recipe["Ingredients"])

st.subheader("🥗 Nutritional Information")
cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["Calories", "Protein", "Carbohydrates", "Fats"],
    [
        f"{selected_recipe['Calories']:.2f} kcal",
        f"{selected_recipe['Protein']:.2f} g",
        f"{selected_recipe['Carbohydrates']:.2f} g",
        f"{selected_recipe['Fats']:.2f} g"
    ]
):
    col.metric(label, value)
cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["Fibre", "Sugar", "Sodium", "Vitamin C"],
    [
        f"{selected_recipe['Fibre']:.2f} g",
        f"{selected_recipe['Sugar']:.2f} g",
        f"{selected_recipe['Sodium']:.2f} mg",
        f"{selected_recipe['Vitamin_C']:.2f} mg"
    ]
):
    col.metric(label, value)

st.divider()

# ============================================================
# RATE RECIPE: build real collaborative data over time
# ============================================================
st.header("⭐ Rate a Recipe (Collaborative Filtering)")
st.caption("Ratings from multiple users help the recommender learn shared preferences.")
with st.form("rating_form"):
    user_id = st.text_input("Choose a user ID (e.g., user_01)", value="user_01")
    rating_value = st.slider("Your rating for the selected recipe", 1, 5, 4)
    submit_rating = st.form_submit_button("Save Rating")

if submit_rating:
    if not user_id.strip():
        st.error("Please enter a user ID.")
    else:
        new_row = pd.DataFrame([{
            "user_id": user_id.strip(),
            "recipe": selected_recipe_name,
            "rating": int(rating_value)
        }])
        if os.path.exists(RATINGS_FILE):
            try:
                existing = pd.read_csv(RATINGS_FILE)
                existing.columns = [str(c).strip().lower() for c in existing.columns]
                if {"user_id", "recipe", "rating"}.issubset(existing.columns):
                    existing = existing[~(
                        (existing["user_id"].astype(str) == user_id.strip()) &
                        (existing["recipe"].astype(str) == selected_recipe_name)
                    )]
                    saved = pd.concat([existing, new_row], ignore_index=True)
                else:
                    saved = new_row
            except Exception:
                saved = new_row
        else:
            saved = new_row
        try:
            saved.to_csv(RATINGS_FILE, index=False)
            st.success("Rating saved to ratings.csv. Refresh/re-run to update collaborative scores.")
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save rating: {exc}")

if cf_available:
    st.success(
        f"Collaborative filtering is active using {ratings_df['user_id'].nunique()} users "
        f"and {ratings_df['recipe'].nunique()} rated recipes."
    )
else:
    st.info(
        "Collaborative filtering is waiting for ratings from at least 2 different users "
        "across at least 2 recipes. Add ratings using the form, or provide ratings.csv. "
        "Until then, the collaborative component stays neutral rather than inventing ratings."
    )

with st.expander("How to prepare ratings.csv"):
    st.write("Place `ratings.csv` beside `app.py` with these columns: `user_id`, `recipe`, `rating`.")
    st.dataframe(pd.DataFrame([
        {"user_id": "user_01", "recipe": df.iloc[0]["Recipe"], "rating": 5},
        {"user_id": "user_02", "recipe": df.iloc[min(1, len(df)-1)]["Recipe"], "rating": 4},
        {"user_id": "user_01", "recipe": df.iloc[min(1, len(df)-1)]["Recipe"], "rating": 4},
    ]), hide_index=True, use_container_width=True)
    st.caption("These are format examples only, not real user ratings. Replace them with genuine ratings.")

# ============================================================
# COMBINED RECOMMENDATIONS: content + collaborative + GI
# ============================================================
st.divider()
st.header("✨ Hybrid Recommendations: Collaborative Filtering + GI Scoring")
st.write(
    "The ranking combines recipe similarity, collaborative preference signals (when real ratings "
    "are available), and a relative lower-GI ingredient score."
)
if st.button("🔍 Find Diabetes-Aware Smart Alternatives", type="primary", use_container_width=True):
    rows = []
    for idx, row in df.iterrows():
        if idx == recipe_index:
            continue
        gi_score, mean_gi, gi_count = recipe_gi_score(row["Ingredients"])
        content_score = float(np.clip(hybrid_similarity[recipe_index, idx], 0, 1))
        cf_score = float(cf_scores[idx]) if cf_available else 0.5
        # With no ratings, CF is neutral and the UI clearly says so.
        combined = 0.50 * content_score + 0.25 * cf_score + 0.25 * gi_score
        rows.append({
            "index": idx,
            "Recipe": row["Recipe"],
            "Content similarity": content_score,
            "Collaborative score": cf_score,
            "Relative lower-GI score": gi_score,
            "Matched GI ingredients": gi_count,
            "Approx. mean ingredient GI": mean_gi,
            "Combined score": combined
        })
    ranked = pd.DataFrame(rows).sort_values("Combined score", ascending=False).head(5)

    st.subheader("Top recommended recipes")
    st.caption(
        "Scores are ranking aids, not medical ratings. A recipe's mean ingredient GI is not "
        "the measured GI of the complete meal."
    )
    for rank, (_, result) in enumerate(ranked.iterrows(), start=1):
        item = df.iloc[int(result["index"])]
        with st.container(border=True):
            st.subheader(f"{rank}. {item['Recipe']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Combined score", f"{result['Combined score']:.3f}")
            c2.metric("Content similarity", f"{result['Content similarity']:.3f}")
            c3.metric("Relative lower-GI score", f"{result['Relative lower-GI score']:.2f}")
            if cf_available:
                st.write(f"**Collaborative preference score:** {result['Collaborative score']:.2f}")
            else:
                st.write("**Collaborative preference score:** neutral (not enough real ratings yet)")
            if pd.notna(result["Approx. mean ingredient GI"]):
                st.write(f"**Approx. mean GI of matched ingredients:** {result['Approx. mean ingredient GI']:.1f}")
            else:
                st.write("**GI information:** not enough ingredient values in the reference table.")
            st.write("**Ingredients:**", item["Ingredients"])
            a, b, c = st.columns(3)
            a.metric("Calories", f"{item['Calories']:.1f} kcal")
            b.metric("Carbohydrates", f"{item['Carbohydrates']:.1f} g")
            c.metric("Fibre", f"{item['Fibre']:.1f} g")

    # ========================================================
    # GI-AWARE INGREDIENT SWAPS, PERSONALIZED BY CF SIGNAL
    # ========================================================
    st.divider()
    st.header("🔄 GI-aware Ingredient Swaps")
    st.write(
        "Candidate swaps are ranked using approximate GI reference values and, when available, "
        "collaborative ratings for recipes containing the replacement. Check that the substitute "
        "fits the dish; a lower GI alone does not guarantee a healthier meal."
    )

    ingredient_text = normalize_text(selected_recipe["Ingredients"])
    swap_rows = []
    matched_originals = set()

    # For each known substitution, check if the source ingredient appears.
    for source, (replacement, reason) in SUBSTITUTES.items():
        if source not in ingredient_text:
            continue
        matched_originals.add(source)
        source_gi = ingredient_gi(source)
        replacement_gi = ingredient_gi(replacement)
        # Some multi-option replacements are not in the GI table; use first named option.
        replacement_key = replacement.split(" or ")[0].split(",")[0].strip()
        replacement_gi = ingredient_gi(replacement_key)
        if source_gi is None or replacement_gi is None:
            gi_improvement = None
            gi_score = 0.5
        else:
            gi_improvement = source_gi - replacement_gi
            gi_score = float(np.clip((50 + gi_improvement) / 100, 0, 1))

        replacement_norm = normalize_text(replacement_key)
        # Collaborative signal from recipes whose ingredient text contains replacement.
        candidate_recipe_indices = [
            i for i, ing in enumerate(df["Ingredients"].astype(str))
            if replacement_norm and replacement_norm in normalize_text(ing)
        ]
        if cf_available and candidate_recipe_indices:
            collab_score = float(np.mean(cf_scores[candidate_recipe_indices]))
        else:
            collab_score = 0.5

        combined_swap_score = 0.65 * gi_score + 0.35 * collab_score
        swap_rows.append({
            "Ingredient to replace": source.title(),
            "Suggested alternative": replacement,
            "Approx. source GI": source_gi,
            "Approx. alternative GI": replacement_gi,
            "GI change (source - alternative)": gi_improvement,
            "Collaborative preference": collab_score if cf_available else np.nan,
            "Swap ranking score": combined_swap_score,
            "Reason": reason
        })

    if swap_rows:
        swaps_df = pd.DataFrame(swap_rows).sort_values("Swap ranking score", ascending=False)
        st.dataframe(swaps_df, use_container_width=True, hide_index=True)
        if not cf_available:
            st.info(
                "Swap ranking currently uses GI score plus a neutral collaborative component. "
                "As genuine ratings from multiple users are collected, collaborative preference "
                "will influence the ranking."
            )
    else:
        st.info(
            "No ingredients matched the built-in swap dictionary for this recipe. "
            "You can extend SUBSTITUTES in app.py with additional ingredients."
        )

    # ========================================================
    # NUTRITION COMPARISON + DISTRIBUTION
    # ========================================================
    st.divider()
    st.header("📈 Nutrition Comparison")
    comparison = pd.concat([
        selected_recipe.to_frame().T,
        df.loc[ranked["index"].astype(int).tolist()]
    ], ignore_index=True)
    comparison_names = comparison["Recipe"].tolist()
    x = np.arange(len(comparison_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, comparison["Protein"].astype(float), width, label="Protein (g)")
    ax.bar(x + width/2, comparison["Fibre"].astype(float), width, label="Fibre (g)")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_names, rotation=25, ha="right")
    ax.set_ylabel("Amount (g)")
    ax.set_title("Protein and Fibre Comparison")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Similarity score distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(hybrid_similarity[recipe_index], bins=15)
    ax2.set_xlabel("Content + nutrition similarity")
    ax2.set_ylabel("Number of recipes")
    ax2.set_title("Recipe similarity score distribution")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()
st.caption(
    "GI values are approximate reference values and may not exist for every ingredient. "
    "This app does not calculate a clinical GI or predict an individual's blood-glucose response. "
    "People with diabetes should follow individualized advice from a qualified clinician/dietitian."
)
