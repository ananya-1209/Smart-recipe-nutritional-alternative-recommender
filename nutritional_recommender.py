import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# SMART RECIPE & NUTRITIONAL ALTERNATIVE RECOMMENDER
# KAGGLE DATASET VERSION
# =========================================================

import os

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data.csv"
)

print("\n" + "=" * 65)
print("       SMART RECIPE & NUTRITIONAL ALTERNATIVE RECOMMENDER")
print("                    KAGGLE DATASET")
print("=" * 65)


# =========================================================
# 1. LOAD KAGGLE DATASET
# =========================================================

print("\nLoading Kaggle dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# =========================================================
# 2. SELECT IMPORTANT COLUMNS
# =========================================================

columns_needed = [
    "final_food_name",
    "Cleaned-Ingredients",
    "Calories (kcal)",
    "Carbohydrates (g)",
    "Protein (g)",
    "Fats (g)",
    "Free Sugar (g)",
    "Fibre (g)",
    "Sodium (mg)",
    "Calcium (mg)",
    "Iron (mg)",
    "Vitamin C (mg)",
    "Folate (µg)"
]

df = df[columns_needed].copy()


# =========================================================
# 3. RENAME COLUMNS
# =========================================================

df.rename(
    columns={
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
    },
    inplace=True
)


# =========================================================
# 4. DATA CLEANING
# =========================================================

print("\nCleaning dataset...")

# Remove rows without recipe or ingredients
df.dropna(
    subset=["Recipe", "Ingredients"],
    inplace=True
)

# Convert nutrition columns to numeric
nutrition_columns = [
    "Calories",
    "Carbohydrates",
    "Protein",
    "Fats",
    "Sugar",
    "Fibre",
    "Sodium",
    "Calcium",
    "Iron",
    "Vitamin_C",
    "Folate"
]

for column in nutrition_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Fill missing nutrition values with median
for column in nutrition_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )

# Remove duplicate recipes
df.drop_duplicates(
    subset=["Recipe"],
    inplace=True
)

# Reset index
df.reset_index(drop=True, inplace=True)

print("Cleaned dataset shape:", df.shape)


# =========================================================
# 5. SHOW AVAILABLE RECIPES
# =========================================================

print("\nAvailable Recipes:")

for recipe in df["Recipe"].head(30):

    print("-", recipe)

print(
    "\nShowing first 30 recipes from the Kaggle dataset."
)


# =========================================================
# 6. INGREDIENT SUBSTITUTION DATABASE
# =========================================================

substitutes = {

    "paneer": (
        "Tofu",
        "Plant-based protein alternative"
    ),

    "rice": (
        "Quinoa",
        "Higher protein and fibre"
    ),

    "milk": (
        "Almond Milk",
        "Dairy-free alternative"
    ),

    "ghee": (
        "Cold-Pressed Sesame Oil",
        "Source of healthy fats"
    ),

    "sugar": (
        "Jaggery",
        "Alternative natural sweetener"
    ),

    "banana": (
        "Apple",
        "Provides fibre and variety"
    ),

    "tomato": (
        "Red Bell Pepper",
        "Rich in vitamin C"
    ),

    "cucumber": (
        "Zucchini",
        "Hydrating vegetable alternative"
    ),

    "moong": (
        "Masoor Dal",
        "Protein-rich alternative"
    ),

    "dal": (
        "Mixed Lentils",
        "Adds nutritional variety"
    ),

    "turmeric": (
        "Fresh Turmeric",
        "Contains antioxidant compounds"
    ),

    "potato": (
        "Sweet Potato",
        "Higher fibre alternative"
    ),

    "cream": (
        "Greek Yogurt",
        "Protein-rich alternative"
    )
}


# =========================================================
# 7. TF-IDF INGREDIENT SIMILARITY
# =========================================================

print("\nCalculating ingredient similarity...")

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    df["Ingredients"].astype(str)
)

ingredient_similarity = cosine_similarity(
    tfidf_matrix
)


# =========================================================
# 8. NUTRITION SIMILARITY
# =========================================================

print("Calculating nutritional similarity...")

nutrition = df[nutrition_columns]

scaler = MinMaxScaler()

nutrition_scaled = scaler.fit_transform(
    nutrition
)

nutrition_similarity = cosine_similarity(
    nutrition_scaled
)


# =========================================================
# 9. HYBRID SIMILARITY
# =========================================================

print("Calculating hybrid similarity...")

hybrid_similarity = (

    0.7 * ingredient_similarity

    +

    0.3 * nutrition_similarity

)


# =========================================================
# 10. USER INPUT
# =========================================================

recipe_name = input(
    "\nEnter recipe name: "
).strip()


# Case-insensitive matching
matches = df[
    df["Recipe"].str.lower()
    == recipe_name.lower()
]


if matches.empty:

    print("\nRecipe not found.")

    print(
        "\nPlease enter a recipe name exactly as "
        "shown in the available recipes."
    )

    exit()


recipe_index = matches.index[0]


# =========================================================
# 11. SELECTED RECIPE
# =========================================================

selected_recipe = df.iloc[recipe_index]


print("\n" + "=" * 45)
print("SELECTED RECIPE")
print("=" * 45)

print(
    "Recipe:",
    selected_recipe["Recipe"]
)

print(
    "Ingredients:",
    selected_recipe["Ingredients"]
)

print(
    "Calories:",
    round(selected_recipe["Calories"], 2)
)

print(
    "Protein:",
    round(selected_recipe["Protein"], 2),
    "g"
)

print(
    "Carbohydrates:",
    round(selected_recipe["Carbohydrates"], 2),
    "g"
)

print(
    "Fats:",
    round(selected_recipe["Fats"], 2),
    "g"
)

print(
    "Fibre:",
    round(selected_recipe["Fibre"], 2),
    "g"
)

print(
    "Sugar:",
    round(selected_recipe["Sugar"], 2),
    "g"
)


# =========================================================
# 12. TOP RECOMMENDED RECIPES
# =========================================================

scores = list(
    enumerate(
        hybrid_similarity[recipe_index]
    )
)

scores = sorted(
    scores,
    key=lambda x: x[1],
    reverse=True
)


print("\n" + "=" * 45)
print("TOP RECOMMENDED RECIPES")
print("=" * 45)


recommendation_count = 0


for index, score in scores:

    if index == recipe_index:
        continue

    recipe = df.iloc[index]

    print(
        "\nRecipe:",
        recipe["Recipe"]
    )

    print(
        "Similarity Score:",
        round(score, 3)
    )

    print(
        "Calories:",
        round(recipe["Calories"], 2)
    )

    print(
        "Protein:",
        round(recipe["Protein"], 2),
        "g"
    )

    print(
        "Carbohydrates:",
        round(recipe["Carbohydrates"], 2),
        "g"
    )

    print(
        "Fats:",
        round(recipe["Fats"], 2),
        "g"
    )

    print(
        "Fibre:",
        round(recipe["Fibre"], 2),
        "g"
    )

    recommendation_count += 1

    if recommendation_count == 3:
        break


# =========================================================
# 13. SMART INGREDIENT SUBSTITUTIONS
# =========================================================

print("\n" + "=" * 45)
print("SMART INGREDIENT SUBSTITUTIONS")
print("=" * 45)


ingredients = (
    selected_recipe["Ingredients"]
    .lower()
    .replace(",", " ")
    .split()
)


sub_found = False


for ingredient in ingredients:

    if ingredient in substitutes:

        replacement, benefit = substitutes[
            ingredient
        ]

        print(
            f"\n{ingredient.title()} --> {replacement}"
        )

        print(
            "Benefit:",
            benefit
        )

        sub_found = True


if not sub_found:

    print(
        "No predefined ingredient substitutions "
        "available for this recipe."
    )


# =========================================================
# 14. NUTRITIONAL WELLNESS SUGGESTION
# =========================================================

print("\n" + "=" * 45)
print("NUTRITIONAL WELLNESS INFORMATION")
print("=" * 45)


if selected_recipe["Calories"] > 400:

    print(
        "This recipe has relatively high calories. "
        "Consider balancing it with lighter meals."
    )

else:

    print(
        "This recipe has a moderate calorie value."
    )


if selected_recipe["Protein"] >= 12:

    print(
        "Good protein content."
    )

else:

    print(
        "Consider adding another protein-rich food."
    )


if selected_recipe["Fibre"] >= 5:

    print(
        "Good fibre content."
    )

else:

    print(
        "Consider adding more fibre-rich foods."
    )


# =========================================================
# 15. SIMILARITY SCORE DISTRIBUTION GRAPH
# =========================================================

all_scores = hybrid_similarity[
    recipe_index
]

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    all_scores,
    bins=15
)

plt.title(
    "Recipe Similarity Score Distribution"
)

plt.xlabel(
    "Hybrid Similarity Score"
)

plt.ylabel(
    "Number of Recipes"
)

plt.tight_layout()

plt.savefig(
    "similarity_distribution.png",
    dpi=300
)

plt.close()

print(
    "\nSaved: similarity_distribution.png"
)


# =========================================================
# 16. NUTRITION COMPARISON GRAPH
# =========================================================

top_indices = []

recommendation_count = 0


for index, score in scores:

    if index == recipe_index:
        continue

    top_indices.append(index)

    recommendation_count += 1

    if recommendation_count == 3:
        break


comparison_indices = [
    recipe_index
] + top_indices


comparison_names = [
    df.iloc[i]["Recipe"]
    for i in comparison_indices
]


comparison_protein = [
    df.iloc[i]["Protein"]
    for i in comparison_indices
]


comparison_fibre = [
    df.iloc[i]["Fibre"]
    for i in comparison_indices
]


x = range(
    len(comparison_names)
)


width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    [i - width / 2 for i in x],
    comparison_protein,
    width,
    label="Protein (g)"
)


plt.bar(
    [i + width / 2 for i in x],
    comparison_fibre,
    width,
    label="Fibre (g)"
)


plt.xticks(
    list(x),
    comparison_names,
    rotation=25,
    ha="right"
)


plt.title(
    "Protein and Fibre Comparison"
)

plt.ylabel(
    "Amount (g)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "nutrition_comparison.png",
    dpi=300
)

plt.close()

print(
    "Saved: nutrition_comparison.png"
)


# =========================================================
# 17. SIMULATED USER SATISFACTION
# =========================================================

system_a = [
    70,
    74,
    76,
    75,
    77
]

system_b = [
    80,
    84,
    82,
    85,
    88
]


avg_a = sum(system_a) / len(
    system_a
)

avg_b = sum(system_b) / len(
    system_b
)


print("\n" + "=" * 45)
print("SIMULATED A/B TESTING")
print("=" * 45)


print(
    "System A Average Satisfaction:",
    round(avg_a, 2)
)

print(
    "System B Average Satisfaction:",
    round(avg_b, 2)
)


if avg_b > avg_a:

    print(
        "Result: Hybrid Recommendation System "
        "performs better."
    )

else:

    print(
        "Result: Original Recommendation System "
        "performs better."
    )


# =========================================================
# 18. USER SATISFACTION GRAPH
# =========================================================

plt.figure(
    figsize=(6, 5)
)


plt.bar(
    ["System A", "System B"],
    [avg_a, avg_b]
)


plt.title(
    "User Satisfaction Comparison"
)

plt.ylabel(
    "Average Satisfaction Score"
)

plt.tight_layout()

plt.savefig(
    "user_satisfaction_comparison.png",
    dpi=300
)

plt.close()


print(
    "Saved: user_satisfaction_comparison.png"
)


# =========================================================
# 19. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("              PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nDataset used:")
print(
    "Indian_Food_Ingrediants_Nutrition_CookingMethods.csv"
)

print("\nGenerated graphs:")

print(
    "1. similarity_distribution.png"
)

print(
    "2. nutrition_comparison.png"
)

print(
    "3. user_satisfaction_comparison.png"
)

print("\n===============================================")