import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import csv
import os
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

def append_company_csv(list_info_com):
    file_exist = os.path.isfile("companies.csv")
    if any(not item.strip() for item in list_info_com):
        return "❌ Some fields are empty!"
    with open("companies.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["Name", "Job_type", "Description", "Location"]
        if not file_exist:
            writer.writerow(header)
        writer.writerow(list_info_com)

def append_worker_csv(list_info_worker):
    file_exist = os.path.isfile("workers.csv")
    if any(not item.strip() for item in list_info_worker):
        return "❌ Some fields are empty!"
    with open("workers.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["Name", "Age", "Resume", "Location"]
        if not file_exist:
            writer.writerow(header)
        writer.writerow(list_info_worker)

def find_best_job(resume_text, csv_file_path):
    df = pd.read_csv(csv_file_path)
    job_des = df['Description'].astype(str).tolist()
    resume = [resume_text]
    similarities = vectorizer_similarity(resume, job_des)
    sim_scores = similarities[0]
    sorted_similarities = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

    print("\n🔍 Top 5 Job Matches:")
    for rank, (job_index, score) in enumerate(sorted_similarities[:5], start=1):
        job = df.iloc[job_index]
        print(f"{rank}. {job['Job_type']} at {job['Name']} ({round(score * 100, 2)}% match)")

def graph():
    try:
        df_worker = pd.read_csv("workers.csv")
        df_jobs = pd.read_csv("companies.csv")

        G = nx.Graph()

        workers = [f"👤 {row['Name']}" for _, row in df_worker.iterrows()]
        companies = [f"🏢 {row['Name']} - {row['Job_type']}" for _, row in df_jobs.iterrows()]

        G.add_nodes_from(workers, bipartite=0)
        G.add_nodes_from(companies, bipartite=1)

        for _, w_row in df_worker.iterrows():
            resume_text = [str(w_row['Resume'])]  # Ensure 'Resume' matches column name exactly
            job_desc_list = df_jobs['Description'].astype(str).tolist()
            similarities = vectorizer_similarity(resume_text, job_desc_list)[0]

            for idx, score in enumerate(similarities):
                if score > 0.01:  # Only meaningful edges
                    worker_node = f"👤 {w_row['Name']}"
                    job_node = f"🏢 {df_jobs.iloc[idx]['Name']} - {df_jobs.iloc[idx]['Job_type']}"
                    G.add_edge(worker_node, job_node, weight=round(score, 2))

        pos = {}
        pos.update((node, (1, i)) for i, node in enumerate(workers))
        pos.update((node, (2, i)) for i, node in enumerate(companies))

        plt.figure(figsize=(14, 9))
        edge_weights = nx.get_edge_attributes(G, 'weight')

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=['lightblue' if n in workers else 'lightgreen' for n in G.nodes()],
            node_size=1200,
            font_size=8,
            edge_color='gray'
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()},
            font_color='red'
        )

        plt.title("📊 Resume-Job Matching Graph (Weighted by Similarity)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("⚠️ Error in graph:", e)

# Main CLI
while True:
    topic = input("Enter 1 for company\nEnter 2 for worker\nEnter 3 to see graph\n>>> ")
    if topic == "1":
        com_name = input("Company name: ")
        job_type = input("Job title: ")
        description = input("Job description: ")
        location = input("Location: ")
        append_company_csv([com_name, job_type, description, location])
        print("✅ Company saved.\n")

    elif topic == "2":
        name = input("Your name: ")
        age = input("Your age: ")
        resume = input("Your resume/skills: ")
        location = input("Your location: ")
        append_worker_csv([name, age, resume, location])
        print("✅ Worker saved.\n")
        find_best_job(resume, "companies.csv")

    elif topic == "3":
        graph()
    else:
        print("❌ Invalid option.")
