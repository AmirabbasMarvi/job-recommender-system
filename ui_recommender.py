import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vectorizer_similarity(resume_of_person, description_job):
    model = TfidfVectorizer()
    all_info = resume_of_person + description_job
    all_info_vec = model.fit_transform(all_info)
    resume_vec = all_info_vec[:len(resume_of_person)]
    descript_vec = all_info_vec[len(resume_of_person):]
    similarity = cosine_similarity(resume_vec, descript_vec)
    return similarity

def append_csv(file_path, header, row_data):
    file_exist = os.path.isfile(file_path)
    if any(not item.strip() for item in row_data):
        return False
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exist:
            writer.writerow(header)
        writer.writerow(row_data)
    return True

def find_best_job(resume_text, df_jobs):
    job_des = df_jobs['Description'].astype(str).tolist()
    resume = [resume_text]
    similarities = vectorizer_similarity(resume, job_des)[0]
    sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    return [(df_jobs.iloc[i]['Job_type'], df_jobs.iloc[i]['Name'], score) for i, score in sorted_similarities[:5]]

def draw_graph():
    try:
        df_worker = pd.read_csv("workers.csv")
        df_jobs = pd.read_csv("companies.csv")

        G = nx.Graph()

        workers = [f"👤 {row['Name']}" for _, row in df_worker.iterrows()]
        companies = [f"🏢 {row['Name']} - {row['Job_type']}" for _, row in df_jobs.iterrows()]

        G.add_nodes_from(workers, bipartite=0)
        G.add_nodes_from(companies, bipartite=1)

        for _, w_row in df_worker.iterrows():
            resume_text = [str(w_row['Resume'])]
            similarities = vectorizer_similarity(resume_text, df_jobs['Description'].astype(str).tolist())[0]
            for idx, score in enumerate(similarities):
                if score > 0.01:
                    worker_node = f"👤 {w_row['Name']}"
                    job_node = f"🏢 {df_jobs.iloc[idx]['Name']} - {df_jobs.iloc[idx]['Job_type']}"
                    G.add_edge(worker_node, job_node, weight=round(score, 2))

        pos = {}
        pos.update((node, (1, i)) for i, node in enumerate(workers))
        pos.update((node, (2, i)) for i, node in enumerate(companies))

        plt.figure(figsize=(14, 9))
        edge_weights = nx.get_edge_attributes(G, 'weight')

        nx.draw(
            G, pos, with_labels=True,
            node_color=['lightblue' if n in workers else 'lightgreen' for n in G.nodes()],
            node_size=1200, font_size=8, edge_color='gray'
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()},
            font_color='red'
        )
        plt.title("📊 Resume-Job Graph")
        plt.axis('off')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"⚠️ Error drawing graph: {e}")

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("💼 Resume-Job Matching System")

menu = st.sidebar.selectbox("Select an action", ["Add Company", "Add Worker", "View Graph"])

if menu == "Add Company":
    st.header("🏢 Add Company Info")
    name = st.text_input("Company Name")
    job = st.text_input("Job Title")
    desc = st.text_area("Job Description")
    loc = st.text_input("Location")
    if st.button("✅ Save Company"):
        success = append_csv("companies.csv", ["Name", "Job_type", "Description", "Location"], [name, job, desc, loc])
        if success:
            st.success("Company added.")
        else:
            st.warning("All fields are required.")

elif menu == "Add Worker":
    st.header("👤 Add Worker Info")
    w_name = st.text_input("Your Name")
    w_age = st.number_input("Your Age", min_value=16, max_value=100, step=1)

    w_resume = st.text_area("Your Resume or Skills")
    w_loc = st.text_input("Your Location")
    if st.button("✅ Save Worker"):
        success = append_csv("workers.csv", ["Name", "Age", "Resume", "Location"], [w_name, w_age, w_resume, w_loc])
        if success:
            st.success("Worker added.")
            if os.path.exists("companies.csv"):
                df_jobs = pd.read_csv("companies.csv")
                st.subheader("🔍 Top 5 Job Matches:")
                matches = find_best_job(w_resume, df_jobs)
                for i, (job, comp, score) in enumerate(matches, 1):
                    st.write(f"{i}. **{job}** at *{comp}* — Match: {round(score * 100, 2)}%")
        else:
            st.warning("All fields are required.")

elif menu == "View Graph":
    st.header("📈 Resume vs Job Graph")
    draw_graph()
