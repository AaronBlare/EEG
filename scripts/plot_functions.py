import plotly.graph_objects as go
import plotly.express as px


def add_scatter_trace(fig, x, y, name, color, marker_symbol, mode='markers', size=8):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            showlegend=showlegend,
            name=name,
            mode=mode,
            marker=dict(
                color=color,
                size=size,
                opacity=0.9,
                symbol=marker_symbol,
                line=dict(
                    color='black',
                    width=0.3
                )
            )
        )
    )


def add_scatter_trace_subj(fig, x, y, name, subj, color, marker_symbol, mode='markers', size=8):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            showlegend=showlegend,
            name=name,
            mode=mode,
            marker=dict(
                color=color,
                size=size,
                opacity=0.9,
                symbol=marker_symbol,
                line=dict(
                    color='black',
                    width=0.3
                )
            ),
            legendgroup=f"{subj}",
            legendgrouptitle_text=f"{subj}",
        )
    )


def add_layout(fig, x_label, y_label, title, font_size=25):
    fig.update_layout(
        template="none",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        title=dict(
            text=title,
            font=dict(
                size=font_size
            ),
            y=0.99
        ),
        autosize=True,
        showlegend=True,
        xaxis=get_axis(x_label, font_size, font_size),
        yaxis=get_axis(y_label, font_size, font_size),
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
        mirror="allticks",
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


def legend_layout(fig, x_label, y_label, title):
    fig.update_layout(
        template="none",
        legend=dict(
            itemsizing='constant',
            orientation="h",
            font_size=20,
        ),
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=120,
            t=230,
            pad=0
        ),
        title=dict(
            text=title,
            xref="paper",
            font=dict(
                size=45
            ),
            y=0.99
        ),
        showlegend=True,
        xaxis=get_axis(x_label, 30, 30),
        yaxis=get_axis(y_label, 30, 30),
        autosize=False,
        width=900,
        height=900,
    )


def plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data, num_subjects, x_axis, y_axis, title):
    for movement_id in range(0, len(experiment)):
        movement = experiment[movement_id]
        symbol = marker_symbols[movement_id]
        curr_movement_data = data.loc[data['class_simp'] == movement]
        for subject_id in range(0, num_subjects):
            curr_subject_data = curr_movement_data.loc[curr_movement_data['subject'] == f'S{subject_id}']
            add_scatter_trace_subj(fig,
                                   curr_subject_data[x_axis].values,
                                   curr_subject_data[y_axis].values,
                                   movement.split('_')[0],
                                   f'S{subject_id}',
                                   colors[subject_id],
                                   symbol)
    add_layout(fig, x_axis, y_axis, f'')
    legend_layout(fig, x_axis, y_axis, title)


def plot_scatter_train_val(fig, experiment, marker_symbols, colors, data, x_axis, y_axis, title):
    for subset_id in range(0, len(experiment)):
        subset = experiment[subset_id]
        curr_subset_data = data.loc[data['split'] == subset]
        right_subset_data = curr_subset_data.loc[data['class_simp'].isin(['right_real', 'right_quasi', 'right_im'])]
        left_subset_data = curr_subset_data.loc[data['class_simp'].isin(['left_real', 'left_quasi', 'left_im'])]
        add_scatter_trace(fig,
                          right_subset_data[x_axis].values,
                          right_subset_data[y_axis].values,
                          f"{subset} right",
                          colors[subset_id],
                          marker_symbols[0])
        add_scatter_trace(fig,
                          left_subset_data[x_axis].values,
                          left_subset_data[y_axis].values,
                          f"{subset} left",
                          colors[subset_id],
                          marker_symbols[1])
    add_layout(fig, x_axis, y_axis, title)
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
