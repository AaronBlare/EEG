import plotly.graph_objects as go
import csv


def add_scatter_trace(fig, x, y, name, mode='lines', size=8):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            showlegend=showlegend,
            name=name,
            mode=mode,
            marker=dict(
                size=size,
                opacity=0.7,
                line=dict(
                    width=1
                )
            )
        )
    )


def add_layout(fig, x_label, y_label, title):
    fig.update_layout(
        template="none",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        ),
        title=dict(
            text=title,
            font=dict(
                size=25
            )
        ),
        autosize=True,
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=80,
            t=100,
            pad=0
        ),
        showlegend=True,
        xaxis=get_axis(x_label, 25, 25),
        yaxis=get_axis(y_label, 25, 25),
    )


def get_axis(title, title_size, tick_size):
    axis = dict(
        title=title,
        autorange=True,
        showgrid=True,
        zeroline=False,
        linecolor='black',
        showline=True,
        gridcolor='gainsboro',
        gridwidth=0.05,
        mirror=True,
        ticks='outside',
        titlefont=dict(
            color='black',
            size=title_size
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            color='black',
            size=tick_size
        ),
        exponentformat='e',
        showexponent='all'
    )
    return axis


def save_figure(fig, fn, path):
    fig.write_image(f"{path}/{fn}.png")
    fig.write_image(f"{path}/{fn}.pdf")


def plot_catboost_evolution(figure_path):

    data_path = 'D:/EEG/'
    x = []
    y_learn = []
    y_test = []
    with open(data_path + "catboost_info/learn_error.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            x.append(int(line[0]))
            y_learn.append(float(line[1]))
    with open(data_path + "catboost_info/test_error.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            y_test.append(float(line[1]))

    fig = go.Figure()
    add_scatter_trace(fig, x, y_learn, "Train")
    add_scatter_trace(fig, x, y_test, "Test")
    add_layout(fig, "Epoch", 'Error', "")
    fig.update_layout({'colorway': ['blue', 'red']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=75,
            t=45,
            pad=0
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=[0, max(y_learn + y_test) + 0.1])
    save_figure(fig, "evolution", figure_path)


def plot_xgboost_evolution(progress, figure_path):

    if 'logloss' in progress['train']:
        flag = 'logloss'
    else:
        flag = 'mlogloss'

    x = list(range(0, len(progress['train'][flag])))
    y_learn = progress['train'][flag]
    y_test = progress['val'][flag]

    fig = go.Figure()
    add_scatter_trace(fig, x, y_learn, "Train")
    add_scatter_trace(fig, x, y_test, "Test")
    add_layout(fig, "Epoch", 'Error', "")
    fig.update_layout({'colorway': ['blue', 'red']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=75,
            t=45,
            pad=0
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=[0, max(y_learn + y_test) + 0.1])
    save_figure(fig, "evolution", figure_path)
