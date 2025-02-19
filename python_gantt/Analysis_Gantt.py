import plotly.express as px
import pandas as pd

# Add vertical line to indicate a deadline
def deadline(date, label):
    fig.add_shape( type="line", x0=date, x1=date, y0=0, y1=1,  line=dict(color="red", width=2, dash="dash"),  xref="x", yref="paper")
    fig.add_annotation(x=date, y=1.05, text=label, showarrow=False, xref="x", yref="paper",font=dict(color="red", size=12),align="center",)

def dependency(date_from, task_from, date_to, task_to):
    fig.add_shape(type="line", x0=date_from, y0=task_from, x1=date_to, y1=task_to, 
                  line=dict(color="black", width=2, dash="solid"), xref="x", yref="y",)

df = pd.DataFrame([

    dict(Task="Flight", Start='2025-05-01', Finish='2025-07-30', Resource="Collab.", Label='1st test arrays & beam map'),
    dict(Task="Flight", Start='2026-04-01', Finish='2026-04-30', Resource="Collab.", Label='Packing<br>planning<br>&org'),
    dict(Task="Flight", Start='2026-06-01', Finish='2026-06-30', Resource="Collab.", Label='Palestine<br>integration'),
    dict(Task="Flight", Start='2026-08-01', Finish='2026-08-30', Resource="Collab.", Label='Ship MCM'),
    dict(Task="Flight", Start='2026-07-01', Finish='2026-08-30', Resource="Collab.", Label='Personal prep.'),
    dict(Task="Flight", Start='2026-11-01', Finish='2026-11-30', Resource="Collab.", Label='Arrive MCM'),
    dict(Task="Flight", Start='2026-12-01', Finish='2026-12-30', Resource="Collab.", Label='Declare<br>flight ready'),
    dict(Task="Flight", Start='2027-01-01', Finish='2027-01-30', Resource="Collab.", Label='FLIGHT'),
    dict(Task="Flight", Start='2027-02-01', Finish='2027-02-28', Resource="Collab.", Label='Pack<br>&leave'),

    dict(Task="Publication", Start='2027-04-01', Finish='2027-06-30', Resource="Collab.", Label='Decide series<br>papers'),
    dict(Task="Publication", Start='2027-11-01', Finish='2028-01-30', Resource="Collab.", Label='Writing<br>time'),
    dict(Task="Publication", Start='2028-02-15', Finish='2028-03-30', Resource="Collab.", Label='Collab.<br>Review'),
    dict(Task="Publication", Start='2028-04-01', Finish='2028-05-30', Resource="Collab.", Label='Published<br>papers'),
    dict(Task="Publication", Start='2028-06-01', Finish='2028-07-30', Resource="Collab.", Label='LIM28'),   

    dict(Task="Milestones", Start='2026-07-01', Finish='2026-09-30', Resource="Collab.", Label='decide on QA / Null tests'),
    dict(Task="Milestones", Start='2027-03-01', Finish='2027-05-30', Resource="Collab.", Label='Data reduction'),
    dict(Task="Milestones", Start='2027-06-01', Finish='2027-06-30', Resource="Collab.", Label='QA'),
    dict(Task="Milestones", Start='2027-07-01', Finish='2027-07-15', Resource="Collab.", Label='1st CII map'),
    dict(Task="Milestones", Start='2027-07-15', Finish='2027-07-30', Resource="Collab.", Label='Push pipeline button'),
    dict(Task="Milestones", Start='2025-02-01', Finish='2025-02-28', Resource="Collab.", Label='Discuss<br>in-flight<br>plots'),

    dict(Task="Forecast suite", Start='2026-09-01', Finish='2027-03-30', Resource="R,K,M", Label='forecasts available on repo'),
    dict(Task="Forecast suite", Start='2025-02-01', Finish='2025-02-15', Resource="R,K,M", Label='Mock CII 3D cubes on repo'),
    dict(Task="Forecast suite", Start='2025-02-15', Finish='2025-02-28', Resource="R,K,M", Label='Mock galaxy cubes on repo'),
    dict(Task="Forecast suite", Start='2025-02-15', Finish='2025-02-28', Resource="R,K,M", Label='Timestream on repo'),
    dict(Task="Forecast suite", Start='2025-02-15', Finish='2025-02-28', Resource="R,K,M", Label='Noise realisation on repo'),
    dict(Task="Forecast suite", Start='2025-04-01', Finish='2025-05-30', Resource="R,K,M", Label='Every thing on repo'),


    dict(Task="Map Making", Start='2025-04-01', Finish='2025-11-01', Resource="M", Label='Dev. Mapmaker'),
    dict(Task="Map Making", Start='2026-03-01', Finish='2026-03-30', Resource="M", Label='Push Mapmaker'),
    dict(Task="Map Making", Start='2024-11-01', Finish='2024-11-30', Resource="M", Label='read<br>other<br>data<br>format'),
    dict(Task="Map Making", Start='2024-12-01', Finish='2025-01-30', Resource="M", Label='Implement<br>freqs'),
    dict(Task="Map Making", Start='2025-02-01', Finish='2025-03-15', Resource="M", Label='Namap<br>ready'),

    dict(Task="Map Making", Start='2027-8-01', Finish='2027-9-30', Resource="M", Label='1st<br>[CII]<br>maps'),
    dict(Task="Map Making", Start='2027-10-01', Finish='2027-10-30', Resource="M", Label='Push Analysis button'),

    dict(Task="Scan Strategy", Start='2025-01-01', Finish='2025-03-30', Resource="F", Label='Publication scan and noise'),
    dict(Task="Scan Strategy", Start='2026-01-01', Finish='2026-03-30', Resource="?", Label='Scan codes on repo'),
    dict(Task="Scan Strategy", Start='2025-11-01', Finish='2025-12-30', Resource="?", Label='In flight plan ready'),

    dict(Task="Pointed Obs.", Start='2024-11-01', Finish='2025-01-30', Resource="M,D", Label='Forecast<br>integration<br>time'),
    dict(Task="Pointed Obs.", Start='2025-02-01', Finish='2025-02-28', Resource="M,D", Label='CO<br>z=0<br>targets'),
    dict(Task="Pointed Obs.", Start='2025-06-01', Finish='2025-07-30', Resource="M,D", Label='Review+<br>&Publication'),
    dict(Task="Pointed Obs.", Start='2025-09-01', Finish='2025-10-30', Resource="M,D", Label='In flight plan'),

    dict(Task="Cross-power", Start='2024-11-01', Finish='2025-01-30', Resource="J,R", Label='Paper publication'),
    dict(Task="Cross-power", Start='2026-03-01', Finish='2026-03-30', Resource="J,R", Label='Pipeline on repo'),
    dict(Task="Cross-power", Start='2026-01-01', Finish='2026-02-15', Resource="J,R", Label='Test on mock'),
    dict(Task="Cross-power", Start='2026-07-01', Finish='2026-07-30', Resource="J,R", Label='Euclid'),
    dict(Task="Cross-power", Start='2026-08-01', Finish='2026-10-30', Resource="J,R", Label='Build Euclid cube'),
    dict(Task="Cross-power", Start='2026-11-01', Finish='2026-11-15', Resource="J,R", Label='Euclid on repo'),

    dict(Task="Auto-power", Start='2026-03-01', Finish='2026-03-30', Resource="R,J,S", Label='Pipeline on repo'),

    dict(Task="Masking", Start='2025-01-01', Finish='2025-01-30', Resource="?", Label='Make SIDES Interlopers map'),
    dict(Task="Masking", Start='2025-02-01', Finish='2025-03-30', Resource="?", Label='Mask simu'),
    dict(Task="Masking", Start='2025-04-01', Finish='2025-05-30', Resource="?", Label='Make Astrodeep mask'),
    dict(Task="Masking", Start='2025-09-01', Finish='2025-09-30', Resource="?", Label='Mask on repo'),

    dict(Task="Calibration", Start='2025-07-01', Finish='2025-09-30', Resource="E,F,M", Label='Calibration strategy dev'),
    dict(Task="Calibration", Start='2026-03-01', Finish='2026-03-30', Resource="E,F,M", Label='Plan in repo'),

    dict(Task="Stacking", Start='2025-01-01', Finish='2025-02-28', Resource="S", Label='Paper publication'),
    dict(Task="Stacking", Start='2026-01-01', Finish='2026-03-30', Resource="S", Label='Pipeline on repo'),
    
    dict(Task="CIB sep.", Start='2024-11-01', Finish='2024-11-08', Resource="M,Srini", Label='SIDES CIB cubes ready'),
    dict(Task="CIB sep.", Start='2024-11-01', Finish='2024-11-08', Resource="M,Srini", Label="Measure R$\\nu \\nu'$"),
    dict(Task="CIB sep.", Start='2025-07-31', Finish='2025-07-31', Resource="M,Srini", Label="Publication"),

    dict(Task="Blind Detection", Start='2026-03-01', Finish='2026-03-30', Resource="?", Label='Pipeline on repo'),
   
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource", text='Label')

#-- Deadlines --

deadline('2027-10-01','Push Analysis button')
deadline('2026-04-01','Start preparing flight')
deadline('2028-09-01','End<br>of<br>Grant')


#-- Dependencies --

fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.update_xaxes(title='', range=['2024-10-01', '2028-09-30'])
fig.update_traces( textposition="inside", texttemplate="%{text}", insidetextanchor="middle", textangle=0,textfont=dict(size=8) )
fig.update_layout(margin=dict(t=30, l=20, b=20, r=20), height=400 )
fig.show()

