import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame([

    dict(Task="Cryo", Start='2025-02-01', Finish='2025-02-28', Resource="Illinois", Label = 'Filters<br>arrive'),
    dict(Task="Cryo", Start='2025-04-01', Finish='2025-04-30', Resource="Illinois", Label = 'Cryostat<br>optical<br>tests'),
    dict(Task="Cryo", Start='2025-06-01', Finish='2025-06-30', Resource="Illinois", Label = 'Test<br>array(s)<br>arrive'),
    dict(Task="Cryo", Start='2025-07-01', Finish='2025-07-30', Resource="Illinois", Label = 'Near<br>field<br>beam<br>mapping'),
    dict(Task="Cryo", Start='2025-08-01', Finish='2025-08-30', Resource="Illinois", Label = 'FTS<br>bandpass'),
    dict(Task="Cryo", Start='2025-09-01', Finish='2025-09-30', Resource="Illinois", Label = 'cryo<br>sloshing<br>thermal<br>stability'),
    dict(Task="Cryo", Start='2026-02-01', Finish='2026-02-28', Resource="Illinois", Label = 'Ship<br>cryostat<br>to UAZ'),

    dict(Task="Detectors", Start='2025-09-01', Finish='2025-09-30', Resource="JPL", Label='RF<br>Starlink<br>tests'),
    dict(Task="Detectors", Start='2025-12-01', Finish='2025-12-30', Resource="JPL", Label='Flight<br>arrays<br>delivered'),
    dict(Task="Detectors", Start='2026-01-01', Finish='2026-01-30', Resource="JPL", Label='Final<br>integrated<br>optical<br>tests'),
    dict(Task="Detectors", Start='2026-03-01', Finish='2026-03-30', Resource="JPL", Label='Full<br>system<br>integration<br>tests'),
    dict(Task="Detectors", Start='2026-05-01', Finish='2026-05-30', Resource="JPL", Label='Pack<br>from<br>UAZ to PST'),

    dict(Task="Readout", Start='2024-11-01', Finish='2024-11-30', Resource="Software<br>Team", Label='decision<br>on<br>RFSoC<br>vs<br>T0'),
    dict(Task="Readout", Start='2026-08-01', Finish='2026-12-30', Resource="Software<br>Team", Label='Data<br>aquisition<br>training?'),

    dict(Task="Gondola", Start='2024-11-01', Finish='2024-11-30', Resource="Penn", Label='IF design'),
    dict(Task="Gondola", Start='2024-11-01', Finish='2024-11-30', Resource="Penn", Label='IF fab?'),
    dict(Task="Gondola", Start='2025-01-01', Finish='2025-01-30', Resource="Penn", Label='Scoop<br>design'),
    dict(Task="Gondola", Start='2025-02-01', Finish='2025-02-28', Resource="Penn", Label='gondola<br>postmortem?'),
    dict(Task="Gondola", Start='2025-03-01', Finish='2025-03-30', Resource="Penn", Label='IF<br>interface<br>with<br>OF<br>mirror'),
    dict(Task="Gondola", Start='2025-04-01', Finish='2025-04-30', Resource="Penn", Label='Mirror to SC<br>alignment<br>(IR camera)'),

    dict(Task="General", Start='2026-04-01', Finish='2026-04-30', Resource="All", Label='Packing<br>planning<br>&org'),
    dict(Task="General", Start='2026-06-01', Finish='2026-06-30', Resource="All", Label='Palestine<br>integration'),
    dict(Task="General", Start='2026-08-01', Finish='2026-08-30', Resource="All", Label='Ship<br>MCM'),
    dict(Task="General", Start='2026-07-01', Finish='2026-08-30', Resource="All", Label='Personal<br>prep.'),
    dict(Task="General", Start='2026-11-01', Finish='2026-11-30', Resource="All", Label='Arrive<br>MCM'),
    dict(Task="General", Start='2026-12-01', Finish='2026-12-30', Resource="All", Label='Declare<br>flight<br>ready'),
    dict(Task="General", Start='2027-01-01', Finish='2027-01-30', Resource="All", Label='FLIGHT'),
    dict(Task="General", Start='2027-02-01', Finish='2027-02-28', Resource="All", Label='Pack<br>&leave'),
    #dict(Task="General", Start='2028-09-01', Finish='2028-09-30', Resource="All", Label='End of Grant'),
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource", text='Label')
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
# Set tighter x-axis limits if needed
fig.update_xaxes(title='', range=['2024-10-01', '2027-03-01'])

# Show, center, and increase the font size of labels inside the bars
fig.update_traces(textposition="inside", texttemplate="%{text}", insidetextanchor="middle",textangle=0,
                  textfont=dict(size=10))  # Increase font size to 14

# Update layout for compact view
fig.update_layout(margin=dict(t=30, l=30, b=30, r=30), height=600)


fig.show()
plt.show()

