import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import re
from collections import Counter
#creating the dataset

# File path to the downloaded JSON file
file_path = '/Users/ansgarkamratowski/Desktop/Unsupervised learn/arxiv-metadata-oai-snapshot.json'

data = []

print("Starting to read the JSON file...")
with jsonlines.open(file_path) as reader:
    for i, obj in enumerate(reader):
        if i % 10000 == 0:
            print(f"Processing record {i}")
        data.append({
            'id': obj.get('id'),
            'submitter': obj.get('submitter'),
            'authors': obj.get('authors'),  # The raw authors string
            'authors_parsed': obj.get('authors_parsed'),  # Extracting the parsed authors field
            'title': obj.get('title'),
            'abstract': obj.get('abstract'),
            'categories': obj.get('categories'),
            'journal-ref': obj.get('journal-ref'),
            'doi': obj.get('doi'),
            'date': obj.get('update_date')
        })


print("Finished reading the JSON file!")

# Convert the list to pandas DataFrame for easier processing
df = pd.DataFrame(data)
print(df.columns)
print("Finished loading preliminary dataframe")
##Add higher Groupings 
prefix_to_group = {
    'cs': 'Computer_Science',
    'econ': 'Economics',
    'eess': 'Electrical_Engineering_and_Systems_Science',
    'math': 'Mathematics',
    'astro-ph': 'Physics',
    'cond-mat': 'Physics',
    'gr-qc': 'Physics',
    'hep': 'Physics',
    'nlin': 'Physics',
    'nucl': 'Physics',
    'physics': 'Physics',
    'quant-ph': 'Physics',
    'math-ph': 'Physics',
    'q-bio': 'Quantitative_Biology',
    'q-fin': 'Quantitative_Finance',
    'stat': 'Statistics'
}
def assign_group(category):
    for prefix, group in prefix_to_group.items():
        if category.startswith(prefix):
            return group
    return 'Other'

def process_groups(df):
    df['group'] = df['categories'].apply(lambda x: ' '.join(sorted(set(assign_group(cat) for cat in x.split()))))
    return df

df = process_groups(df)
df['date'] = pd.to_datetime(df['date'])
print("added groups to data frame")
print(df[['categories', 'group']].head(30))

# Display the first few rows of the DataFrame
print(df.head())
print(df.columns)
#check unique groups
unique_groups = pd.Series(df['group'].apply(lambda x: x.split()).explode().unique())
print("Unique individual groups:")
print(unique_groups)


#Basic metrics

##Number of articles
num_articles = len(df)
##articles per year
articles_per_year = pd.to_datetime(df['date']).dt.year.value_counts().sort_index()
print(articles_per_year)


##Unique categories
unique_categories = pd.Series(df['categories'].apply(lambda x: x.split()).explode().unique())
print("Unique individual categories:")
print(unique_categories)

#Groups
exploded_groups = df['group'].apply(lambda x: x.split()).explode()
group_counts = exploded_groups.value_counts()
##percentages of groups per year
df_exploded = df.assign(group=df['group'].str.split()).explode('group')
papers_per_year_group = df_exploded.groupby([pd.to_datetime(df_exploded['date']).dt.year, 'group']).size().unstack(fill_value=0)
papers_per_year_group_percent = papers_per_year_group.div(papers_per_year_group.sum(axis=1), axis=0) * 100

##Crossdisiplinary groups
df['num_groups'] = df['group'].apply(lambda x: len(x.split()))
group_count_distribution = df['num_groups'].value_counts().sort_index()
print(group_count_distribution)
group_count_percentages = (group_count_distribution / num_articles) * 100
print(group_count_percentages)
cross_group_count = df.groupby([pd.to_datetime(df['date']).dt.year, 'num_groups']).size().unstack(fill_value=0)
cross_group_percentage = cross_group_count.div(cross_group_count.sum(axis=1), axis=0) * 100


def weighted_co_occurrence_matrix_categories(df):
    # Split the categories into a list for each row
    df['categories_list'] = df['categories'].apply(lambda x: x.split())
    
    # Step 1: Count overall frequency of each category
    category_frequencies = Counter(c for categories in df['categories_list'] for c in categories)
    
    # Step 2: Count co-occurrences (as before)
    co_occurrences = Counter()
    for categories in df['categories_list']:
        category_count = len(categories)
        if category_count > 1:
            weight = 1 / category_count
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    co_occurrences[(categories[i], categories[j])] += weight
                    co_occurrences[(categories[j], categories[i])] += weight
    
    # Step 3: Create the co-occurrence matrix
    unique_categories = sorted(set(c for categories in df['categories_list'] for c in categories))
    co_matrix = pd.DataFrame(0, index=unique_categories, columns=unique_categories)
    
    # Step 4: Fill in the co-occurrence matrix
    for (category1, category2), count in co_occurrences.items():
        co_matrix.at[category1, category2] = count
        co_matrix.at[category2, category1] = count
    
    # Step 5: Normalise the co-occurrence matrix by category frequencies
    for category1 in unique_categories:
        for category2 in unique_categories:
            if category1 != category2 and co_matrix.at[category1, category2] > 0:
                co_matrix.at[category1, category2] = co_matrix.at[category1, category2] / np.sqrt(category_frequencies[category1] * category_frequencies[category2])
    
    return co_matrix

# Apply the function to the DataFrame
categories_co_matrix = weighted_co_occurrence_matrix_categories(df)

# Display the co-occurrence matrix
print(categories_co_matrix)

def weighted_co_occurrence_matrix_group(df):
    df['groups_list'] = df['group'].apply(lambda x: x.split())
    
    co_occurrences = Counter()
    
    # Step 1: Count overall frequency of each group
    group_frequencies = Counter(g for groups in df['groups_list'] for g in groups)
    
    # Step 2: Count co-occurrences
    for groups in df['groups_list']:
        group_count = len(groups)
        if group_count > 1:
            weight = 1 / group_count
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    co_occurrences[(groups[i], groups[j])] += weight
                    co_occurrences[(groups[j], groups[i])] += weight
    
    unique_groups = sorted(set(g for groups in df['groups_list'] for g in groups))
    co_matrix = pd.DataFrame(0, index=unique_groups, columns=unique_groups)
    
    # Step 3: Fill in the co-occurrence matrix
    for (group1, group2), count in co_occurrences.items():
        co_matrix.at[group1, group2] = count
        co_matrix.at[group2, group1] = count

    # Step 4: Normalise the co-occurrence matrix by group frequencies
    for group1 in unique_groups:
        for group2 in unique_groups:
            if group1 != group2 and co_matrix.at[group1, group2] > 0:
                # Normalise the co-occurrence count
                co_matrix.at[group1, group2] = co_matrix.at[group1, group2] / np.sqrt(group_frequencies[group1] * group_frequencies[group2])
    
    return co_matrix
groups_co_matrix = weighted_co_occurrence_matrix_group(df)
print(groups_co_matrix)

def rank_co_occurrences(co_matrix):
    co_occurrence_rankings = co_matrix.stack().reset_index()
    
    # Rename the columns for clarity
    co_occurrence_rankings.columns = ['Group 1', 'Group 2', 'Co-occurrence']
    
    # Step 2: Filter out self-pairs (where Group 1 == Group 2)
    co_occurrence_rankings = co_occurrence_rankings[co_occurrence_rankings['Group 1'] != co_occurrence_rankings['Group 2']]
    
    # Step 3: Handle symmetry by only keeping one version of each pair (e.g., only keep Group 1 < Group 2)
    co_occurrence_rankings = co_occurrence_rankings[co_occurrence_rankings['Group 1'] < co_occurrence_rankings['Group 2']]
    
    # Step 4: Sort by co-occurrence value in descending order
    co_occurrence_rankings = co_occurrence_rankings.sort_values(by='Co-occurrence', ascending=False)
    
    return co_occurrence_rankings
co_occurrence_rankings = rank_co_occurrences(groups_co_matrix)
print(co_occurrence_rankings.head(36))
co_occurrence_rankings.to_csv('/Users/ansgarkamratowski/Desktop/co_occurrence_rankings.csv', index=False)

def unweighted_co_occurrence_matrix(df):
    # Step 1: Split the 'group' column and explode into separate rows
    df['groups_list'] = df['group'].apply(lambda x: x.split())
    
    # Step 2: Initialize an empty Counter for co-occurrences
    co_occurrences = Counter()
    
    # Step 3: Loop over each article and count group co-occurrences
    for groups in df['groups_list']:
        if len(groups) > 1:
            # No weighting here: Simply count each co-occurrence
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    co_occurrences[(groups[i], groups[j])] += 1
                    co_occurrences[(groups[j], groups[i])] += 1
    
    # Step 4: Create a matrix of unique groups
    unique_groups = sorted(set(g for groups in df['groups_list'] for g in groups))
    co_matrix = pd.DataFrame(0, index=unique_groups, columns=unique_groups)
    
    # Step 5: Fill in the unweighted co-occurrence matrix
    for (group1, group2), count in co_occurrences.items():
        co_matrix.at[group1, group2] = count
        co_matrix.at[group2, group1] = count
    
    return co_matrix
unweighted_co_matrix = unweighted_co_occurrence_matrix(df)
print(unweighted_co_matrix)
unweighted_co_matrix.to_csv('/Users/ansgarkamratowski/Desktop/unweighted_co_occurrence_matrix.csv')

##Authors
def format_author_name(author_info):
    last_name = author_info[0].strip()
    first_name = author_info[1].strip()
    return f"{first_name} {last_name}".strip()

df['author_list'] = df['authors_parsed'].apply(lambda authors: [format_author_name(author) for author in authors])
exploded_authors = df.explode('author_list')
author_counts = exploded_authors['author_list'].value_counts()
unique_authors = exploded_authors['author_list'].unique()
authors_per_paper = exploded_authors.groupby('id')['author_list'].count()
# Calculate the overall average number of authors per paper
average_authors_overall = authors_per_paper.mean()
median_authors_overall = authors_per_paper.median()

# group specific
df['num_authors_per_group'] = [
    num_authors / len(groups) if len(groups) > 0 else 0  # Avoid division by zero
    for num_authors, groups in zip(df['num_authors'], df['group_list'])
]
mean_authors_per_group = df_exploded.groupby('group_list')['num_authors_per_group'].mean()
median_authors_per_group = df_exploded.groupby('group_list')['num_authors_per_group'].median()


df['num_authors'] = df.groupby('id')['author_list'].transform('count')
average_authors_per_group = df.groupby('group')['num_authors'].mean()
median_authors_per_group = df.groupby('group')['num_authors'].median()

print("Average number of authors per group-paper:")
print(average_authors_per_group)
print(median_authors_per_group)

print("median and mean of authors:")
print(average_authors_overall)
print(median_authors_overall)

print("Unique authors:")
print(unique_authors)
print("Authors with the most citations:")
print(author_counts.head(10))

#plots
# Plots author count per paper
plt.figure(figsize=(10, 6))
plt.hist(authors_per_paper, bins=range(1, 32), edgecolor='black', color='mediumturquoise')
plt.title('Distribution of Author Counts per Paper (Limited to 30 Authors)', fontsize=16)
plt.xlabel('Number of Authors', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(ticks=range(1, 31), rotation=90)
plt.show()

# Plots authors for groups
for group in df_exploded['group_list'].unique():
    # Filter the DataFrame for the current group
    group_df = df_exploded[df_exploded['group_list'] == group]
    
    # Step 2: Plot the histogram for the distribution of 'num_authors_per_group' for the current group
    plt.figure(figsize=(10, 6))
    plt.hist(group_df['num_authors_per_group'], bins=range(1, 32), edgecolor='black',       color='mediumturquoise')
    
    # Step 3: Add titles and labels
    plt.title(f'Distribution of Authors per Group for {group}', fontsize=14)
    plt.xlabel('Number of Authors per Paper', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Step 4: Display the plot
    plt.show()

# Plots a bar chart of the number of articles per year
plt.figure(figsize=(10, 6))
plt.bar(articles_per_year.index, articles_per_year.values, color='mediumturquoise')
plt.title('Number of Articles Published Per Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.xticks(ticks=articles_per_year.index, rotation=90)
plt.show()

# Plots a bar chart of the number of cross disiplinary groups e.g. economics and physics will be 2
plt.figure(figsize=(10, 6))
plt.bar(group_count_distribution.index, group_count_distribution.values, color='mediumturquoise')
plt.title('Number of Cross-Disciplinary Papers by Number of Groups', fontsize=16)
plt.xlabel('Number of Groups', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.show()

# Plots a bar chart percentage of Articles Belonging to Each Group
total_group_assignments = group_counts.sum()
group_percentages = (group_counts / total_group_assignments) * 100
plt.figure(figsize=(10, 6))
plt.bar(group_percentages.index, group_percentages.values, color='mediumturquoise')
plt.title('Percentage of Articles Belonging to Each Group (Based on Group Assignments)', fontsize=16)
plt.xlabel('', fontsize=12)
plt.ylabel('Percentage of Group Assignments (%)', fontsize=12)
plt.xticks(rotation=90, ha='right')
plt.show()

# plots stacked percentage bar 
plt.figure(figsize=(12, 8))
papers_per_year_group_percent.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
plt.title('Percentage of Papers Belonging to Each Group Per Year (Stacked)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Papers (%)', fontsize=12)
plt.xticks(rotation=90)
plt.legend(title="Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# plots stacked bar 
plt.figure(figsize=(12, 8))
papers_per_year_group.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
plt.title('Number of Papers Belonging to Each Group Per Year (Stacked)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(rotation=90)
plt.legend(title="Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# plots stacked cross group collabs per year
plt.figure(figsize=(12, 8))
cross_group_count.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
plt.title('Number of Cross-Group Collaborations Per Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(rotation=90)
plt.legend(title="Number of Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# plots stacked cross group percentage collabs per year
plt.figure(figsize=(12, 8))
cross_group_percentage.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
plt.title('Percentage of Cross-Group Collaborations Per Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Papers (%)', fontsize=12)
plt.xticks(rotation=90)
plt.legend(title="Number of Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#creating network graphs
def build_graph(co_matrix):
    G = nx.Graph()
    for i in range(len(co_matrix)):
        for j in range(i + 1, len(co_matrix)):
            weight = co_matrix.iloc[i, j]
            if weight > 0:
                G.add_edge(co_matrix.index[i], co_matrix.index[j], weight=weight)
    return G

def filter_above_threshold(G, threshold):
    H = nx.Graph()
    for edge in G.edges(data=True):
        if edge[2]['weight'] > threshold:
            H.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
    return H

def filter_below_threshold(G, threshold):
    H = nx.Graph()
    for edge in G.edges(data=True):
        if edge[2]['weight'] <= threshold:
            H.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
    return H

def top_3_connections(G, order=True):
    H = nx.Graph()
    for node in G.nodes():
        neighbors = [(neighbor, G[node][neighbor]['weight']) for neighbor in G.neighbors(node)]
        top_3_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=order)[:3]
        for neighbor, weight in top_3_neighbors:
            H.add_edge(node, neighbor, weight=weight)
    return H

def draw_graph(H, title):
    pos = nx.spring_layout(H, seed=42)
    weights = [edge[2]['weight'] for edge in H.edges(data=True)]
    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.cm.RdYlGn
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    nx.draw_networkx_nodes(H, pos, node_size=700, node_color="mediumturquoise", ax=ax)
    edge_colors = [cmap(norm(weight)) for weight in weights]
    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, edge_cmap=cmap, width=2, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=10, ax=ax)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    plt.colorbar(sm, ax=ax)
    plt.title(title)
    plt.show()

def main(co_matrix, matrix_type='Co-occurrence'):
    G = build_graph(co_matrix)
    weights = [edge[2]['weight'] for edge in G.edges(data=True)]
    mean_weight = np.mean(weights)
    median_weight = np.median(weights)
    
    H_all_connections = G
    H_above_mean = filter_above_threshold(G, mean_weight)
    H_below_mean = filter_below_threshold(G, mean_weight)
    H_above_median = filter_above_threshold(G, median_weight)
    H_below_median = filter_below_threshold(G, median_weight)
    H_top_3 = top_3_connections(G,True)
    H_bottom_3 = top_3_connections(G, False)
    
    draw_graph(H_all_connections, f'{matrix_type}: All Connections')
    draw_graph(H_above_mean, f'{matrix_type}: Connections Above Mean Threshold')
    draw_graph(H_below_mean, f'{matrix_type}: Connections Below Mean Threshold')
    draw_graph(H_above_median, f'{matrix_type}: Connections Above Median Threshold')
    draw_graph(H_below_median, f'{matrix_type}: Connections Below Median Threshold')
    draw_graph(H_top_3, f'{matrix_type}: Top 3 Strongest Connections')
    draw_graph(H_bottom_3, f'{matrix_type}: Top 3 weakest Connections')

main(groups_co_matrix, matrix_type='Group')

def filter_by_group(co_matrix, group):
    # Find categories that belong to the given group
    selected_categories = [cat for cat in co_matrix.index if any(cat.startswith(prefix) for prefix, grp in prefix_to_group.items() if grp == group)]
    
    # Filter the co-occurrence matrix to only include the selected categories
    filtered_matrix = co_matrix.loc[selected_categories, selected_categories]
    
    return filtered_matrix

groups = ['Computer_Science', 'Economics', 'Electrical_Engineering_and_Systems_Science', 
          'Mathematics', 'Physics', 'Quantitative_Biology', 'Quantitative_Finance', 'Statistics']

def top_intra(co_matrix, top_n):
    combined_df = pd.DataFrame()

    for group in groups:
        filtered_matrix = filter_by_group(co_matrix, group)
        ranked_df = rank_co_occurrences(filtered_matrix)
        top_10_df = ranked_df.head(top_n)
        
        top_10_df = top_10_df.reset_index()
        top_10_df['Group'] = group
        
        combined_df = pd.concat([combined_df, top_10_df], ignore_index=True)

    return combined_df
def bottom_intra(co_matrix, bottom_n):
    combined_df = pd.DataFrame()

    for group in groups:
        filtered_matrix = filter_by_group(co_matrix, group)
        ranked_df = rank_co_occurrences(filtered_matrix) 
        top_10_df = ranked_df.tail(bottom_n)
        
        top_10_df = top_10_df.reset_index()
        top_10_df['Group'] = group
        
        combined_df = pd.concat([combined_df, top_10_df], ignore_index=True)
    return combined_df

top_3 = top_intra(categories_co_matrix,3)
