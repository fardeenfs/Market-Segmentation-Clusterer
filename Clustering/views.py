import json
from datetime import datetime

from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
import ast
from Clustering.forms import ScatterPlotGenerate


def scatter_plot_generate(uid, data, field1, field2):
    # Set the styles to Seaborn
    sns.set()

    # We will use KMeans function of the sklearn module to cluster our data
    from sklearn.cluster import KMeans

    # Generating a scatter plot.
    plt.scatter(data[:, 0], data[:, 1])

    # Variables being considered for the clustering
    plt.xlabel(field1)
    plt.ylabel(field2)
    dir = 'static/'
    plot_img = 'scatter_plot_' + uid + '.png'
    plt.savefig(dir + plot_img)
    return plot_img


def find_elbow(uid, max_clusters, scaled_data):
    sumofsquares_cluster_graph = []
    if max_clusters > 100:
        max_clusters = 100

    # We find the sum of squares from 1 cluster to max_clusters clusters.
    for i in range(1, max_clusters):
        kmeans = KMeans(i)
        kmeans.fit(scaled_data)
        sumofsquares_cluster_graph.append(kmeans.inertia_)
    plt.clf()
    plt.plot(range(1, max_clusters), sumofsquares_cluster_graph)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squares')
    plot_img = 'elbow_graph_' + uid + '.png'
    dir = 'static/'
    plt.savefig(dir + plot_img)
    return plot_img


def k_clusterer(uid, k, scaled_data,field1,field2):
    plt.clf()
    kmeans = KMeans(k)
    kmeans.fit(scaled_data)
    clusters = scaled_data.copy()
    print(clusters)
    clusters['cluster_pred'] = kmeans.fit_predict(scaled_data)
    plt.scatter(clusters['0'], clusters['1'], c=clusters['cluster_pred'], cmap='rainbow')
    plt.xlabel(field1)
    plt.ylabel(field2)
    dir = 'static/'
    plot_img = 'cluster_plot_' + uid + '.png'
    plt.savefig(dir + plot_img)
    return plot_img


def data_upload(request):
    uid = str((datetime.now()).strftime("%Y%m%d"))
    if request.method == 'POST':
        field1 = request.POST['field1']
        field2 = request.POST['field2']
        datafile = request.FILES['file']
        raw_data = pd.read_csv(datafile)
        raw_data = raw_data[[field1, field2]]
        data = preprocessing.scale(raw_data)
        print(data)
        plot = scatter_plot_generate(uid, data, field1, field2)
        pd.DataFrame(data).to_csv("scaled_data" + uid + '.csv', index=False)
        return render(request, 'index.html',
                      {'scatterplot': plot, 'ctrl': 2, 'uid': uid, "field1": field1, "field2": field2})
    return render(request, 'index.html', {'ctrl': 1})


def get_elbow_graph(request):
    if request.method == 'POST':
        field1 = request.POST['field1']
        field2 = request.POST['field2']
        uid = request.POST['uid']
        clusters = int(request.POST['max_clusters'])
        data = pd.read_csv("scaled_data" + uid + '.csv')
        plot = find_elbow(uid, clusters, data)
        return render(request, 'index.html', {'plot': plot, 'ctrl': 3, 'uid': uid, "field1": field1, "field2": field2})


def get_clusters(request):
    if request.method == 'POST':
        field1 = request.POST['field1']
        field2 = request.POST['field2']
        uid = request.POST['uid']
        clusters = int(request.POST['max_clusters'])
        data = pd.read_csv("scaled_data" + uid + '.csv')
        plot = k_clusterer(uid, clusters, data,field1,field2)
        return render(request, 'index.html', {'plot': plot, 'ctrl': 4, 'uid': uid, "field1": field1, "field2": field2})
