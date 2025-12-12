import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from config import Settings

# Versuch mplsoccer zu importieren
try:
    from mplsoccer import Pitch
except ImportError:
    class Pitch:
        def __init__(self, *args, **kwargs): pass
        def draw(self, *args, **kwargs): return plt.subplots()
        def bin_statistic(self, *args, **kwargs): return {}
        def heatmap(self, *args, **kwargs): pass

s = Settings()

def plot_zone_analysis(df, pitch_length, pitch_width, team1_name="Team 1", team2_name="Team 2"):
    """
    Erstellt ein Dashboard mit 3 Analysen:
    1. Angriffs-Seiten (Wo greifen sie an?)
    2. Zone Control Map (Wer kontrolliert welchen Bereich?)
    3. Ballbesitz-Verteilung pro Drittel (Wo hält sich das Team auf?)
    
    RICHTUNG: 
    - Team 1 spielt von RECHTS nach LINKS (Ziel: x=0).
    - Team 2 spielt von LINKS nach RECHTS (Ziel: x=100).
    """
    
    # Setup Figure (3 Spalten)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.set_facecolor('#22312b')
    
    # Pitch Instanz
    pitch = Pitch(
        pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
        line_color='#c7d5cc', pitch_color='#22312b', linewidth=2,
    )

    # ---------------------------------------------------------
    # 1. ANGRIFFS-SEITEN (Attack Sides)
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor('#22312b')
    pitch.draw(ax=ax1)
    ax1.invert_yaxis()
    
    # Filter für das jeweilige Angriffsdrittel
    # Team 1 greift auf x=0 an -> Angriffsdrittel ist x < length/3
    t1_att = df[(df["team_ball_control"] == 1) & (df["ball_x_m"] < pitch_length / 3)]
    
    # Team 2 greift auf x=100 an -> Angriffsdrittel ist x > length*2/3
    t2_att = df[(df["team_ball_control"] == 2) & (df["ball_x_m"] > pitch_length * 2/3)]
    
    def get_flank_pct(data, team_direction):
        if len(data) == 0: return [0, 0, 0]
        
        # Y-Achse: 0 ist OBEN, Width ist UNTEN.
        # Zone Oben (y < W/3), Mitte, Zone Unten (y > 2W/3)
        top = len(data[data["ball_y_m"] < pitch_width/3])
        mid = len(data[(data["ball_y_m"] >= pitch_width/3) & (data["ball_y_m"] <= pitch_width*2/3)])
        bot = len(data[data["ball_y_m"] > pitch_width*2/3])
        total = top + mid + bot
        
        # Mapping auf Links/Rechts aus SPIELERSICHT
        if team_direction == "R_to_L": # Team 1 (läuft nach links)
            # Oben (y=0) ist RECHTS
            # Unten (y=60) ist LINKS
            # Return Order: [Links, Mitte, Rechts]
            return [bot/total*100, mid/total*100, top/total*100]
            
        else: # "L_to_R" Team 2 (läuft nach rechts)
            # Oben (y=0) ist LINKS
            # Unten (y=60) ist RECHTS
            return [top/total*100, mid/total*100, bot/total*100]

    # Team 1: Rechts -> Links
    pct1 = get_flank_pct(t1_att, "R_to_L")
    # Team 2: Links -> Rechts
    pct2 = get_flank_pct(t2_att, "L_to_R")
    
    # Balken zeichnen
    # Wir nutzen fixierte Y-Positionen für "Links", "Mitte", "Rechts" (Text-Label)
    # Aber wir zeichnen die Balken an der geometrisch richtigen Stelle im Plot
    
    # Team 1 (Blau): Greift nach Links an.
    # Wir zeichnen die Balken auf der LINKEN Seite des Plots (x=10 startend)
    # Linksaußen ist UNTEN im Plot (y groß)
    # Rechtsaußen ist OBEN im Plot (y klein)
    
    # pct1 ist [Links, Mitte, Rechts]
    # Zeichne "Links" Balken UNTEN
    ax1.barh(pitch_width*5/6, pct1[0]/3, left=5, height=6, color='#00bfff', alpha=0.8) # Links (unten)
    ax1.text(5 + pct1[0]/3 + 2, pitch_width*5/6, f"{int(pct1[0])}%", color='#00bfff', va='center', fontweight='bold')
    
    # Zeichne "Mitte" Balken MITTE
    ax1.barh(pitch_width/2, pct1[1]/3, left=5, height=6, color='#00bfff', alpha=0.8)
    ax1.text(5 + pct1[1]/3 + 2, pitch_width/2, f"{int(pct1[1])}%", color='#00bfff', va='center', fontweight='bold')

    # Zeichne "Rechts" Balken OBEN
    ax1.barh(pitch_width/6, pct1[2]/3, left=5, height=6, color='#00bfff', alpha=0.8) # Rechts (oben)
    ax1.text(5 + pct1[2]/3 + 2, pitch_width/6, f"{int(pct1[2])}%", color='#00bfff', va='center', fontweight='bold')


    # Team 2 (Rot): Greift nach Rechts an.
    # Wir zeichnen Balken auf der RECHTEN Seite (x=95 startend nach links wachsend)
    # pct2 ist [Links, Mitte, Rechts]
    
    # Zeichne "Links" Balken OBEN (da für T2 Oben = Links ist)
    ax1.barh(pitch_width/6, -pct2[0]/3, left=95, height=6, color='#dc143c', alpha=0.8)
    ax1.text(95 - pct2[0]/3 - 8, pitch_width/6, f"{int(pct2[0])}%", color='#dc143c', va='center', fontweight='bold')

    # Zeichne "Mitte" Balken
    ax1.barh(pitch_width/2, -pct2[1]/3, left=95, height=6, color='#dc143c', alpha=0.8)
    ax1.text(95 - pct2[1]/3 - 8, pitch_width/2, f"{int(pct2[1])}%", color='#dc143c', va='center', fontweight='bold')
    
    # Zeichne "Rechts" Balken UNTEN (da für T2 Unten = Rechts ist)
    ax1.barh(pitch_width*5/6, -pct2[2]/3, left=95, height=6, color='#dc143c', alpha=0.8)
    ax1.text(95 - pct2[2]/3 - 8, pitch_width*5/6, f"{int(pct2[2])}%", color='#dc143c', va='center', fontweight='bold')


    ax1.set_title("Angriffs-Seiten (Letztes Drittel)", color='white', fontsize=14, fontweight='bold')
    
    # Pfeile für Spielrichtung
    ax1.arrow(40, pitch_width/2, -10, 0, head_width=3, color='#00bfff', alpha=0.5) # T1 nach Links
    ax1.arrow(60, pitch_width/2, 10, 0, head_width=3, color='#dc143c', alpha=0.5)  # T2 nach Rechts


    # ---------------------------------------------------------
    # 2. ZONE CONTROL MAP (Dominanz-Raster)
    # ---------------------------------------------------------
    ax2 = axes[1]
    pitch.draw(ax=ax2)
    ax2.invert_yaxis()
    
    x1 = df.loc[df["team_ball_control"]==1, "ball_x_m"].values
    y1 = df.loc[df["team_ball_control"]==1, "ball_y_m"].values
    x2 = df.loc[df["team_ball_control"]==2, "ball_x_m"].values
    y2 = df.loc[df["team_ball_control"]==2, "ball_y_m"].values
    
    bin_x, bin_y = 6, 4
    stats1 = pitch.bin_statistic(x1, y1, statistic='count', bins=(bin_x, bin_y))
    stats2 = pitch.bin_statistic(x2, y2, statistic='count', bins=(bin_x, bin_y))
    
    count1 = stats1['statistic']
    count2 = stats2['statistic']
    total = count1 + count2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dominance = (count1 - count2) / total
    dominance[np.isnan(dominance)] = 0 

    cmap_div = LinearSegmentedColormap.from_list("dom_cmap", ['#dc143c', '#444444', '#00bfff'], N=100)
    stats1['statistic'] = dominance
    pitch.heatmap(stats1, ax=ax2, cmap=cmap_div, vmin=-0.8, vmax=0.8, edgecolors='#22312b', alpha=0.9)
    
    ax2.set_title("Territorial-Kontrolle (Blau vs Rot)", color='white', fontsize=14, fontweight='bold')


    # ---------------------------------------------------------
    # 3. BALLBESITZ PRO DRITTEL (Verteilung)
    # ---------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor('#22312b')
    
    def get_zone_distribution(team_id):
        sub = df[df["team_ball_control"] == team_id]
        if len(sub) == 0: return [0, 0, 0]
        
        x = sub["ball_x_m"]
        limit1 = pitch_length / 3
        limit2 = 2 * pitch_length / 3
        
        if team_id == 1:
            # Team 1 spielt RECHTS (100) -> LINKS (0)
            # Defensive: x > 66.6
            # Mitte: 33.3 < x < 66.6
            # Angriff: x < 33.3
            defs = len(x[x > limit2])
            mids = len(x[(x >= limit1) & (x <= limit2)])
            atts = len(x[x < limit1])
        else:
            # Team 2 spielt LINKS (0) -> RECHTS (100)
            # Defensive: x < 33.3
            # Mitte: 33.3 < x < 66.6
            # Angriff: x > 66.6
            defs = len(x[x < limit1])
            mids = len(x[(x >= limit1) & (x <= limit2)])
            atts = len(x[x > limit2])
            
        total_z = defs + mids + atts
        if total_z == 0: return [0, 0, 0]
        return [defs/total_z*100, mids/total_z*100, atts/total_z*100]

    dist1 = get_zone_distribution(1)
    dist2 = get_zone_distribution(2)
    
    # Plotten
    labels = ["Defensive", "Mittelfeld", "Angriff"]
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, dist1, width, label=team1_name, color='#00bfff', alpha=0.9)
    rects2 = ax3.bar(x + width/2, dist2, width, label=team2_name, color='#dc143c', alpha=0.9)
    
    ax3.bar_label(rects1, fmt='%.0f%%', padding=3, color='white', fontweight='bold')
    ax3.bar_label(rects2, fmt='%.0f%%', padding=3, color='white', fontweight='bold')
    
    ax3.set_title("Aufenthaltsorte (Eigene Perspektive)", color='white', fontsize=14, fontweight='bold')
    ax3.set_ylabel("Anteil (%)", color='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color='white', fontsize=11)
    ax3.set_ylim(0, 100) 
    
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='y', colors='white')
    
    legend = ax3.legend(facecolor='#22312b', edgecolor='white')
    for text in legend.get_texts(): text.set_color("white")

    plt.tight_layout()
    return fig