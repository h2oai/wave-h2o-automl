from h2o_wave import Q, ui, app, main, data
import pandas as pd


# Table from Pandas dataframe
def table_from_df(df: pd.DataFrame, table_name: str, sortable=False, filterable=False, searchable=False,
                  groupable=False, downloadable=False, height='90%'):
    # Columns for the table
    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=sortable,  # Make column sortable
        filterable=filterable,  # Make column filterable
        searchable=searchable  # Make column searchable
    ) for x in df.columns.values]
    # Rows for the table
    rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in df.iterrows()]
    table = ui.table(name=f'{table_name}',
             rows=rows,
             columns=columns,
             multiple=False,  # Allow multiple row selection
             groupable=groupable,
             height=height,
             downloadable=downloadable)
    return table


# Show progress bar
async def show_progress(q: Q, card_name: str, card_box: str, message: str):
    q.page[card_name] = ui.form_card(box=card_box, items=[ui.progress(message)])
    await q.page.save()


# Show success bar
async def show_success(q: Q, card_name: str, card_box: str, message: str):
    q.page[card_name] = ui.form_card(box=card_box, items=[ui.message_bar('success', message)])
    await q.page.save()
