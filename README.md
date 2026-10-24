#  Smart Recipe & Nutritional Alternative Recommender

## 1. Project Overview

The **Smart Recipe & Nutritional Alternative Recommender** is a machine learning-based web application designed to recommend recipes based on ingredient similarity, nutritional characteristics, user preferences, and glycemic index (GI) information.

The system combines content-based filtering, collaborative filtering, and GI-aware ingredient scoring to help users explore alternative recipes and make more informed food choices.

The application is developed using Python and Streamlit. It provides an interactive interface where users can select recipes, explore nutritional information, rate recipes, receive recommendations, and investigate alternative ingredients.

The project focuses on combining recommendation system techniques with nutrition-aware scoring, particularly for people who want to monitor their carbohydrate intake and make informed dietary choices.

**Important:** This application is an educational and informational tool. It does not provide medical advice or guarantee that a recipe is suitable for an individual with diabetes.

---

## 2. Problem Statement

Choosing suitable recipes can be challenging because users must consider ingredients, nutritional composition, personal preferences, and the potential effects of carbohydrate-containing foods on blood glucose levels.

Traditional recipe applications often recommend food based primarily on popularity, ingredients, or user preferences without combining these factors with nutritional information and glycemic index considerations.

Furthermore, users may find it difficult to identify practical ingredient alternatives when trying to modify a recipe.

Therefore, there is a need for a recipe recommendation system that combines content similarity, collaborative user preferences, and GI-aware scoring to recommend alternative recipes and ingredient substitutions.

---

## 3. Project Objectives

The main objectives of this project are:

1. To develop an interactive recipe recommendation web application.
2. To recommend similar recipes using ingredient and nutritional similarity.
3. To incorporate collaborative filtering based on user ratings.
4. To use glycemic index reference values to help rank potential ingredient substitutions.
5. To combine recommendation scores to rank alternative recipes.
6. To provide nutritional information for selected and recommended recipes.
7. To compare nutritional characteristics of different recipes.
8. To provide a user-friendly interface through Streamlit.
9. To demonstrate how machine learning and nutrition-related information can be integrated into a recommendation system.

---

## 4. Key Features

### 4.1 Recipe Selection

Users can select a recipe from the available dataset and view its ingredients and nutritional information.

### 4.2 Content-Based Recommendation

The system analyses recipe ingredients using TF-IDF vectorization and cosine similarity to identify recipes with similar ingredient descriptions.

Nutritional similarity can also be incorporated into the recommendation score.

### 4.3 Collaborative Filtering

Collaborative filtering uses ratings provided by users to identify preference patterns between recipes.

The system uses recipe-rating information to calculate collaborative preference scores.

Recommendations become more meaningful as more genuine user ratings are collected.

### 4.4 Glycemic Index-Aware Ingredient Swaps

The application uses approximate glycemic index reference values to help rank potential ingredient alternatives.

For example, it may suggest considering quinoa or brown rice instead of white rice, depending on the recipe.

The suitability of a replacement also depends on portion size, total carbohydrates, preparation method, and the nutritional characteristics of the complete meal.

### 4.5 Hybrid Recommendation System

The hybrid approach combines content similarity, collaborative preference scores, and relative lower-GI ingredient scores.

In the implemented recommendation configuration, the combined recipe score is:

**Combined Score = 0.50 × Content Similarity + 0.25 × Collaborative Score + 0.25 × Relative Lower-GI Score**

The weights can be adjusted to experiment with different recommendation strategies.

When sufficient rating information is unavailable, the collaborative component uses a neutral score rather than inventing user preferences.

### 4.6 Nutritional Information

The application displays nutritional information such as:

- Calories
- Carbohydrates
- Protein
- Fats
- Free sugar
- Fibre
- Sodium
- Calcium
- Iron
- Vitamin C
- Folate

### 4.7 Nutrition Comparison

The system can compare nutritional characteristics of selected and recommended recipes using tables and visualizations.

### 4.8 User Ratings

Users can submit recipe ratings. Ratings are stored in `ratings.csv` and can be used by the collaborative filtering component.

---

## 5. System Architecture

The system follows a modular architecture consisting of the following components:

1. **User Interface Layer:** Streamlit provides the interactive web interface.
2. **Data Loading Layer:** Pandas loads recipe, nutritional, and optional rating datasets.
3. **Preprocessing Layer:** Missing values are handled, nutritional columns are converted to numeric values, and recipe information is cleaned.
4. **Content-Based Recommendation Layer:** TF-IDF and cosine similarity measure ingredient similarity.
5. **Nutritional Similarity Layer:** Min-Max scaling normalizes nutritional features before similarity calculations.
6. **Collaborative Filtering Layer:** Recipe-rating data are used to estimate collaborative preference scores.
7. **GI Scoring Layer:** Approximate GI reference values contribute to ingredient-swap and recipe-ranking scores.
8. **Hybrid Ranking Layer:** The available scores are combined to rank alternative recipes.
9. **Visualization Layer:** Streamlit tables, metrics, and Matplotlib charts present the results.

### System Architecture Diagram

```text
             USER
               |
               v
      STREAMLIT WEB INTERFACE
               |
               v
        DATA LOADING LAYER
               |
       +-------+--------+
       |                |
       v                v
   data.csv         ratings.csv
       |                |
       v                v
 DATA PREPROCESSING   COLLABORATIVE
       |              FILTERING
       v                |
 TF-IDF VECTORIZATION  |
       |                |
       v                |
 COSINE SIMILARITY     |
       |                |
       +--------+-------+
                |
                v
       GI REFERENCE DATA
                |
                v
       GI-AWARE SCORING
                |
                v
       HYBRID RECOMMENDATION
                |
                v
     RECIPE & INGREDIENT RANKING
                |
                v
      NUTRITIONAL COMPARISON
                |
                v
       RESULTS IN STREAMLIT
```

---

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, cosine similarity, and feature scaling |
| Matplotlib | Data visualization |
| Regular Expressions (`re`) | Text normalization and ingredient processing |
| CSV | Storage of recipe, rating, and optional GI data |
| GitHub | Source code management and version control |
| Streamlit Community Cloud | Public deployment of the application |

---

## 7. Machine Learning Methodology

### 7.1 Data Preprocessing

The recipe dataset is loaded from `data.csv`.

The preprocessing steps include:

- Selecting required recipe and nutritional columns.
- Removing rows with missing recipe names or ingredient descriptions.
- Converting nutritional columns to numeric data types.
- Handling missing numerical values using median imputation, with a fallback of zero when a column has no valid median.
- Cleaning recipe names.
- Removing duplicate recipe names.

These steps prepare the data for recommendation and similarity calculations.

### 7.2 TF-IDF Vectorization

Term Frequency–Inverse Document Frequency (TF-IDF) converts ingredient descriptions into numerical feature vectors.

It assigns importance to terms based on their occurrence within a recipe and across the collection of recipes.

Recipes with similar ingredient descriptions may receive more similar vector representations.

### 7.3 Cosine Similarity

Cosine similarity measures the similarity between two numerical vectors.

For vectors \(A\) and \(B\):

\[
\text{Cosine Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

The resulting scores are used to identify recipes with similar ingredient descriptions.

### 7.4 Nutritional Similarity

Nutritional features, including calories, carbohydrates, protein, fats, sugar, fibre, and selected micronutrients, are scaled using Min-Max scaling.

Cosine similarity is then applied to the scaled nutritional feature vectors.

The hybrid content similarity score is:

\[
S_{\text{content}} =
0.7S_{\text{ingredient}}+
0.3S_{\text{nutrition}}
\]

This configuration gives greater weight to ingredient similarity while also considering nutritional characteristics.

### 7.5 Collaborative Filtering

Collaborative filtering uses user-rating information to identify relationships between recipes.

The rating data are organized into a user–recipe matrix.

Cosine similarity between recipe rating vectors helps identify recipes that receive similar ratings from users.

The resulting collaborative score is normalized to a range between 0 and 1.

The method depends on having enough genuine ratings. When rating data are insufficient, the collaborative component remains neutral.

### 7.6 Glycemic Index Scoring

The GI scoring component uses approximate reference values for selected ingredients.

The reference values can be supplied through an optional `ingredient_gi.csv` file, which overrides matching entries in the built-in reference table.

For ingredients with available values, a relative lower-GI score can be calculated as:

\[
S_{\text{GI}}=1-\frac{\text{Mean Ingredient GI}}{100}
\]

A higher score indicates a lower mean GI among the matched reference ingredients.

This is only a simplified ranking measure. It is not the measured glycemic index of the complete recipe.

### 7.7 Hybrid Recommendation

The final recipe ranking combines:

- Content similarity
- Collaborative preference
- Relative lower-GI score

The weighted combination allows the system to consider recipe similarity, user preferences, and GI-related information together.

The resulting score is used to rank candidate recipes.

---

## 8. Machine Learning Pipeline

The application follows this pipeline:

**Step 1: Data Collection**

Load recipe names, ingredient descriptions, and nutritional values from the recipe dataset.

**Step 2: Data Preprocessing**

Clean recipe names, handle missing data, and convert numerical columns into suitable formats.

**Step 3: Feature Extraction**

Transform ingredient descriptions into TF-IDF vectors.

**Step 4: Similarity Calculation**

Calculate ingredient and nutritional similarity between recipes.

**Step 5: Rating Analysis**

Load available user ratings and calculate collaborative preference scores.

**Step 6: GI Scoring**

Match available ingredients to GI reference values and calculate relative scores.

**Step 7: Hybrid Ranking**

Combine content, collaborative, and GI scores using configurable weights.

**Step 8: Ingredient Substitution**

Identify supported ingredients in the selected recipe and suggest predefined alternatives.

**Step 9: Result Presentation**

Display recommended recipes, ingredient alternatives, nutritional metrics, and comparison charts in Streamlit.

---

## 9. Core Features and Application Workflow

The general application workflow is:

1. Open the web application.
2. Select a recipe from the dataset.
3. View its ingredients and nutritional information.
4. Rate the selected recipe if desired.
5. Request diabetes-aware alternative recommendations.
6. Review the ranked recipes and their scores.
7. Explore the suggested ingredient substitutions.
8. Compare nutritional characteristics of the alternatives.

The application uses the available data and scoring rules to rank candidates. Recommendations should be interpreted as informational suggestions rather than medical assessments.

---

## 10. Dataset Format

### 10.1 Recipe Dataset: `data.csv`

The main dataset contains recipe names, ingredient descriptions, and nutritional information.

Expected columns:

| Column | Description |
|---|---|
| `final_food_name` | Recipe name |
| `Cleaned-Ingredients` | Ingredient description |
| `Calories (kcal)` | Energy content |
| `Carbohydrates (g)` | Carbohydrate content |
| `Protein (g)` | Protein content |
| `Fats (g)` | Fat content |
| `Free Sugar (g)` | Free sugar content |
| `Fibre (g)` | Dietary fibre |
| `Sodium (mg)` | Sodium content |
| `Calcium (mg)` | Calcium content |
| `Iron (mg)` | Iron content |
| `Vitamin C (mg)` | Vitamin C content |
| `Folate (µg)` | Folate content |

The column names must match those expected by the application.

### 10.2 Ratings Dataset: `ratings.csv`

The ratings dataset stores user feedback.

Required columns:

| Column | Description |
|---|---|
| `user_id` | Identifier for the rating user |
| `recipe` | Recipe name matching an entry in `data.csv` |
| `rating` | Rating from 1 to 5 |

Example:

```csv
user_id,recipe,rating
user_01,Recipe A,5
user_01,Recipe B,4
user_02,Recipe A,3
user_02,Recipe B,5
```

The recipe names in this example are illustrative. Replace them with names present in the actual dataset.

### 10.3 Optional GI Dataset: `ingredient_gi.csv`

This optional dataset extends the built-in GI reference table.

Expected columns:

| Column | Description |
|---|---|
| `ingredient` | Ingredient name |
| `gi` | Reference glycemic index value |

Example:

```csv
ingredient,gi
brown rice,50
quinoa,53
chickpeas,28
```

These example values are illustrative and should be verified against reliable references before being used for research or dietary decisions.

---

## 11. Application Documentation

### Recipe Selection

The user selects a recipe from the available recipe dataset.

### Nutritional Information

The application displays selected nutritional metrics and the ingredient description.

### Recipe Recommendations

The system calculates similarity and ranks candidate recipes.

### Collaborative Filtering

The application reads available ratings and uses them to calculate preference scores.

### GI-Aware Ingredient Swaps

The system matches supported ingredients to a predefined substitution dictionary and ranks possible alternatives using the implemented scoring method.

### Rating Submission

Users can enter a user ID and submit a rating. The rating is saved in `ratings.csv`.

### Visualizations

Matplotlib is used to display nutritional comparisons and recipe similarity distributions.

---

## 12. Installation and Running Locally

### Prerequisites

Install Python and Visual Studio Code.

The project dependencies should be listed in `requirements.txt`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/ananya-1209/Smart-recipe-nutritional-alternative-recommender.git
```

### Step 2: Open the Project Folder

```bash
cd Smart-recipe-nutritional-alternative-recommender
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

On Windows, if the `python` command does not work, try:

```bash
py -m venv venv
```

### Step 4: Install Dependencies

On Windows:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 5: Run the Application

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will display a local address, usually:

```text
http://localhost:8501
```

Open this address in your browser.

---

## 13. Project Structure

```text
Smart-recipe-nutritional-alternative-recommender/
│
├── app.py
│   └── Main Streamlit application and recommendation interface
│
├── data.csv
│   └── Recipe and nutritional dataset
│
├── ratings.csv
│   └── User ratings for collaborative filtering
│
├── ingredient_gi.csv
│   └── Optional GI reference dataset
│
├── requirements.txt
│   └── Python dependency list
│
└── README.md
    └── Project documentation
```

The `ingredient_gi.csv` file is optional when the application includes built-in GI reference values.

---

## 14. Limitations

1. **Limited rating data:** Collaborative filtering requires sufficient genuine user ratings to identify meaningful preference patterns.
2. **Cold-start problem:** New recipes and new users may have insufficient rating information.
3. **Approximate GI values:** The built-in GI reference values are approximate and may not represent every ingredient variety or preparation method.
4. **Ingredient matching:** Ingredient names and descriptions may not always match the substitution dictionary.
5. **Simplified GI scoring:** Averaging matched ingredient GI values does not calculate the GI of a complete meal.
6. **Nutritional data quality:** Recommendation accuracy depends on the quality and completeness of the dataset.
---

## 15. Future Scope

Potential improvements include:

- Integrating a larger, verified GI reference database.
- Improving ingredient recognition using natural language processing.
- Incorporating serving size and total carbohydrate calculations into ingredient-swap ranking.
- Developing more advanced collaborative filtering methods.
- Collecting more genuine user ratings to improve personalization.
- Adding dietary preferences such as vegetarian, vegan, and allergy-aware filtering.
- Incorporating verified nutrition information from trusted sources.
  
---

## 16. Conclusion

The Smart Recipe & Nutritional Alternative Recommender demonstrates how content-based filtering, collaborative filtering, nutritional similarity, and glycemic index reference information can be combined in an interactive recipe recommendation application.

By integrating these components, the project provides a framework for exploring alternative recipes and ingredient substitutions while considering user preferences and nutritional characteristics.

The project also illustrates the practical application of machine learning, data preprocessing, feature extraction, similarity measurement, and web application development.

---

## 17. Disclaimer

This project is intended for educational and informational purposes only.

Glycemic index values and nutritional information can vary according to ingredients, processing, preparation, serving size, and other factors. The application does not diagnose, treat, or prevent diabetes and should not replace individualized guidance from a qualified healthcare professional or registered dietitian.

---

## 18. Author and Repository

**Project:** Smart Recipe & Nutritional Alternative Recommender

**Repository:** [GitHub Project Repository](https://github.com/ananya-1209/Smart-recipe-nutritional-alternative-recommender)

**Technologies:** Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib

**Deployment:** Streamlit Community Cloud
