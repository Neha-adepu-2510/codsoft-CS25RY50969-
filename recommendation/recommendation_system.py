import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    # Change this to switch datasets (movie/book/product)
    data_type = input("What do you want recommendations for? (movie/book/product): ").strip().lower()

    if data_type == "movie":
        return pd.DataFrame({
            'title': ['The Matrix', 'John Wick', 'Inception'],
            'description': [
                'Sci-fi hacker reality war',
                'Hitman revenge action',
                'Dream inside dream mind-bending thriller'
            ]
        })

    elif data_type == "book":
        return pd.DataFrame({
            'title': ['1984', 'Brave New World', 'Fahrenheit 451'],
            'description': [
                'Dystopian totalitarian future surveillance',
                'Futuristic society control and pleasure',
                'Book burning censorship freedom rebellion'
            ]
        })

    elif data_type == "product":
        return pd.DataFrame({
            'title': ['iPhone 14', 'Samsung Galaxy S23', 'OnePlus 11'],
            'description': [
                'Apple smartphone iOS camera performance',
                'Android flagship AMOLED camera battery',
                'Fast charging smooth performance Android'
            ]
        })

    else:
        print("Invalid type. Please choose movie, book, or product.")
        return None

def recommend_items(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_index(title):
        try:
            return df[df['title'].str.lower() == title.lower()].index[0]
        except:
            return None

    def recommend(title, num=3):
        idx = get_index(title)
        if idx is None:
            print(f"Sorry, '{title}' not found.")
            return

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]

        print(f"\nBecause you liked '{df.iloc[idx]['title']}', you might also like:")
        for i, _ in sim_scores:
            print(f" - {df.iloc[i]['title']}")

    print("\nAvailable items:")
    for t in df['title']:
        print(" -", t)

    while True:
        choice = input("\nEnter the title you like (or 'exit' to quit): ")
        if choice.lower() == 'exit':
            break
        recommend(choice)

if _name_ == "_main_":
    df = load_data()
    if df is not None:
        recommend_items(df)