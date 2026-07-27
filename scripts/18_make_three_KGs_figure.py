#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE = Path('/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT')
NETWORKS = {
    'A': {
        'title': 'Pre-2016 dinoflagellate STX semantic knowledge graph',
        'candidates': [
            BASE/'FINAL_WORKSPACE/kg/dino_pre2016_semantic_edges.csv',
            BASE/'FINAL_WORKSPACE/kg/pre2016_dino_semantic_edges.csv',
            BASE/'FINAL_WORKSPACE/kg/dino_pre_2016_semantic_edges.csv',
        ],
        'output_stem': 'pre2016_dino',
    },
    'B': {
        'title': 'Post-2015 dinoflagellate STX semantic knowledge graph',
        'candidates': [
            BASE/'FINAL_WORKSPACE/kg/dino_post2015_semantic_edges.csv',
            BASE/'FINAL_WORKSPACE/kg/post2015_dino_semantic_edges.csv',
            BASE/'FINAL_WORKSPACE/kg/dino_post_2015_semantic_edges.csv',
        ],
        'output_stem': 'post2015_dino',
    },
    'C': {
        'title': 'Cyanobacterial STX semantic knowledge graph',
        'candidates': [
            BASE/'FINAL_WORKSPACE/kg/cyano_all_semantic_edges.csv',
            BASE/'FINAL_WORKSPACE/kg/cyanobacteria_all_semantic_edges.csv',
            BASE/'FINAL_WORKSPACE/kg/cyano_semantic_edges.csv',
        ],
        'output_stem': 'cyano_all',
    },
}

OUT_DIR = BASE/'FINAL_WORKSPACE/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR/'Figure_three_STX_semantic_KGs.png'
OUT_PDF = OUT_DIR/'Figure_three_STX_semantic_KGs.pdf'
OUT_SVG = OUT_DIR/'Figure_three_STX_semantic_KGs.svg'

MIN_EDGE_WEIGHT = 1
TOP_EDGE_FRACTION = 0.22
TOP_NEIGHBORS_PER_NODE = 6
KEEP_ISOLATED_NODES = False
LAYOUT_SEED = 42
SPRING_ITERATIONS = 800
SPRING_K = 1.35
MIN_NODE_SIZE = 90
MAX_NODE_SIZE = 1400
MIN_EDGE_WIDTH = 0.15
MAX_EDGE_WIDTH = 2.0
EDGE_ALPHA = 0.16
LABEL_FONT_SIZE = 6.2
TITLE_FONT_SIZE = 15
PANEL_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 9
FIGSIZE = (20, 17)
PNG_DPI = 600
MIN_LABEL_DEGREE = 0
LABEL_WRAP_WIDTH = 18

ENTITY_COLORS = {
    'TOXIN': '#D95F02',
    'SXT_GENE': '#1F78B4',
    'DINO_TAXON': '#009E73',
    'CYANO_TAXON': '#56B4E9',
    'ENV_FACTOR': '#E69F00',
    'BIOLOGICAL_PROCESS': '#CC79A7',
    'DETECTION_METHOD': '#999999',
    'OTHER': '#BDBDBD',
}
ENTITY_LABELS = {
    'TOXIN': 'Toxin',
    'SXT_GENE': 'sxt gene',
    'DINO_TAXON': 'Dinoflagellate taxon',
    'CYANO_TAXON': 'Cyanobacteria taxon',
    'ENV_FACTOR': 'Environmental factor',
    'BIOLOGICAL_PROCESS': 'Biological process',
    'DETECTION_METHOD': 'Detection method',
    'OTHER': 'Other',
}

def find_input_file(candidates: list[Path], panel: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Could not locate input for panel {panel}.\nSearched:\n{searched}')

def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f'Expected one of {list(candidates)}; available: {df.columns.tolist()}')
    return None

def clean_entity(value: object) -> str:
    if pd.isna(value):
        return ''
    return re.sub(r'\s+', ' ', str(value).strip())

def normalize_type(value: object) -> str:
    if pd.isna(value):
        return 'OTHER'
    text = str(value).strip().upper().replace('-', '_').replace(' ', '_')
    mapping = {
        'TOXIN':'TOXIN','TOXINS':'TOXIN',
        'SXT_GENE':'SXT_GENE','SXT_GENES':'SXT_GENE','GENE':'SXT_GENE',
        'DINO_TAXON':'DINO_TAXON','DINO':'DINO_TAXON','DINOFLAGELLATE':'DINO_TAXON','DINOFLAGELLATE_TAXON':'DINO_TAXON',
        'CYANO_TAXON':'CYANO_TAXON','CYANO':'CYANO_TAXON','CYANOBACTERIA':'CYANO_TAXON','CYANOBACTERIAL_TAXON':'CYANO_TAXON',
        'ENV_FACTOR':'ENV_FACTOR','ENVIRONMENTAL_FACTOR':'ENV_FACTOR','ENVIRONMENT':'ENV_FACTOR',
        'BIOLOGICAL_PROCESS':'BIOLOGICAL_PROCESS','BIO_PROCESS':'BIOLOGICAL_PROCESS','PROCESS':'BIOLOGICAL_PROCESS',
        'DETECTION_METHOD':'DETECTION_METHOD','METHOD':'DETECTION_METHOD','ANALYTICAL_METHOD':'DETECTION_METHOD',
        'OTHER':'OTHER',
    }
    return mapping.get(text, 'OTHER')

def infer_type(entity: str) -> str:
    t = entity.lower().strip()
    if re.search(r'\bsxt[a-z0-9/]*\b', t, flags=re.I) or 'sxt gene' in t:
        return 'SXT_GENE'
    if any(x in t for x in ['saxitoxin','neosaxitoxin','neostx','gonyautoxin','gtx','paralytic shellfish toxin','pst','dcstx']):
        return 'TOXIN'
    if any(x in t for x in ['alexandrium','gymnodinium','pyrodinium','gonyaulax','dinophysis','prorocentrum','karenia','ostreopsis','coolia','centrodinium','dinoflagellate','protoceratium']):
        return 'DINO_TAXON'
    if any(x in t for x in ['cyanobacteria','cyanobacterium','aphanizomenon','anabaena','dolichospermum','raphidiopsis','cylindrospermopsis','microseira','lyngbya','nostoc','planktothrix']):
        return 'CYANO_TAXON'
    if any(x == t or x in t for x in ['temperature','warming','salinity','light','irradiance','nitrate','nitrogen','phosphate','phosphorus','nutrient','climate','ocean warming']):
        return 'ENV_FACTOR'
    if any(x in t for x in ['hplc','mass spectrometry','lc-ms','mouse bioassay','bioassay','chromatography','elisa','pcr']):
        return 'DETECTION_METHOD'
    return 'BIOLOGICAL_PROCESS'

def wrap_label(text: str, width: int = LABEL_WRAP_WIDTH) -> str:
    words = str(text).split()
    lines, current, length = [], [], 0
    for word in words:
        projected = length + len(word) + (1 if current else 0)
        if projected <= width:
            current.append(word); length = projected
        else:
            lines.append(' '.join(current)); current = [word]; length = len(word)
    if current: lines.append(' '.join(current))
    return '\n'.join(lines)

def load_and_standardize(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f'Input is empty: {path}')
    sc = find_column(df, ['source','node1','entity1','from','u'])
    tc = find_column(df, ['target','node2','entity2','to','v'])
    wc = find_column(df, ['weight','edge_weight','cooccurrence_count','co_occurrence_count','count','frequency','n'], required=False)
    stc = find_column(df, ['source_type','entity1_type','node1_type','type_source'], required=False)
    ttc = find_column(df, ['target_type','entity2_type','node2_type','type_target'], required=False)
    work = pd.DataFrame({'source':df[sc].map(clean_entity),'target':df[tc].map(clean_entity)})
    work['weight'] = 1.0 if wc is None else pd.to_numeric(df[wc], errors='coerce').fillna(1.0)
    work['source_type'] = work['source'].map(infer_type) if stc is None else df[stc].map(normalize_type)
    work['target_type'] = work['target'].map(infer_type) if ttc is None else df[ttc].map(normalize_type)
    work = work[(work.source!='') & (work.target!='') & (work.source!=work.target)].copy()
    pairs = work.apply(lambda r: tuple(sorted((r.source,r.target))), axis=1)
    work['u'] = [p[0] for p in pairs]; work['v'] = [p[1] for p in pairs]
    nodes = pd.concat([
        work[['source','source_type']].rename(columns={'source':'node','source_type':'entity_type'}),
        work[['target','target_type']].rename(columns={'target':'node','target_type':'entity_type'})
    ], ignore_index=True)
    nodes = nodes.groupby('node')['entity_type'].agg(lambda x: x[x!='OTHER'].mode().iloc[0] if not x[x!='OTHER'].mode().empty else 'OTHER').reset_index()
    edges = work.groupby(['u','v'], as_index=False).agg(weight=('weight','sum')).rename(columns={'u':'source','v':'target'})
    return edges, nodes

def filter_edges(edges: pd.DataFrame) -> pd.DataFrame:
    f = edges[edges.weight >= MIN_EDGE_WEIGHT].copy()
    if f.empty:
        raise ValueError('No edges remain after filtering.')
    if 0 < TOP_EDGE_FRACTION < 1:
        f = f.nlargest(max(1, math.ceil(len(f)*TOP_EDGE_FRACTION)), 'weight').copy()
    if TOP_NEIGHBORS_PER_NODE and TOP_NEIGHBORS_PER_NODE > 0:
        rows = []
        nodes = pd.unique(pd.concat([f.source,f.target], ignore_index=True))
        for node in nodes:
            inc = f[(f.source==node)|(f.target==node)]
            rows.append(inc.nlargest(TOP_NEIGHBORS_PER_NODE,'weight'))
        f = pd.concat(rows, ignore_index=True).drop_duplicates(['source','target'])
    return f.sort_values(['weight','source','target'], ascending=[False,True,True])

def build_graph(edges: pd.DataFrame, node_types: pd.DataFrame) -> nx.Graph:
    G = nx.Graph(); lookup = dict(zip(node_types.node,node_types.entity_type))
    for r in edges.itertuples(index=False):
        G.add_node(r.source, entity_type=lookup.get(r.source,infer_type(r.source)))
        G.add_node(r.target, entity_type=lookup.get(r.target,infer_type(r.target)))
        G.add_edge(r.source,r.target,weight=float(r.weight))
    if KEEP_ISOLATED_NODES:
        for n,t in lookup.items(): G.add_node(n, entity_type=t)
    return G

def draw_network(ax, G, title, panel, gdmin, gdmax, gewmin, gewmax):
    k = SPRING_K / math.sqrt(max(G.number_of_nodes(),1)/30)
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, weight='weight', k=k, iterations=SPRING_ITERATIONS)
    wdeg = dict(G.degree(weight='weight')); deg = dict(G.degree())
    vals = np.array([wdeg[n] for n in G.nodes()], float)
    lo, hi = np.log1p(gdmin), np.log1p(gdmax)
    sizes = np.full(len(vals),(MIN_NODE_SIZE+MAX_NODE_SIZE)/2) if math.isclose(lo,hi) else MIN_NODE_SIZE+(np.log1p(vals)-lo)*(MAX_NODE_SIZE-MIN_NODE_SIZE)/(hi-lo)
    size_lookup = dict(zip(G.nodes(), sizes))
    for u,v,d in G.edges(data=True):
        w = float(d.get('weight',1))
        width = (MIN_EDGE_WIDTH+MAX_EDGE_WIDTH)/2 if math.isclose(gewmin,gewmax) else MIN_EDGE_WIDTH+(np.log1p(w)-np.log1p(gewmin))*(MAX_EDGE_WIDTH-MIN_EDGE_WIDTH)/(np.log1p(gewmax)-np.log1p(gewmin))
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],width=width,alpha=EDGE_ALPHA,edge_color='#7A7A7A',ax=ax)
    for et,color in ENTITY_COLORS.items():
        ns=[n for n,a in G.nodes(data=True) if a.get('entity_type','OTHER')==et]
        if ns:
            nx.draw_networkx_nodes(G,pos,nodelist=ns,node_size=[size_lookup[n] for n in ns],node_color=color,edgecolors='white',linewidths=.7,alpha=.96,ax=ax)
    for n,(x,y) in pos.items():
        if deg.get(n,0) < MIN_LABEL_DEGREE: continue
        ax.annotate(wrap_label(n),xy=(x,y),xytext=(2.5,2.5),textcoords='offset points',fontsize=LABEL_FONT_SIZE,fontweight='bold' if deg[n]>=8 else 'normal',ha='left',va='bottom',zorder=10)
    ax.set_title(title,fontsize=TITLE_FONT_SIZE,fontweight='bold',pad=12)
    ax.text(.01,.98,f'({panel})',transform=ax.transAxes,fontsize=PANEL_LABEL_FONT_SIZE,fontweight='bold',ha='left',va='top')
    ax.axis('off'); ax.margins(.16)
    return pd.DataFrame({'node':list(G.nodes()),'entity_type':[G.nodes[n].get('entity_type','OTHER') for n in G.nodes()],'degree':[deg.get(n,0) for n in G.nodes()],'weighted_degree':[wdeg.get(n,0.0) for n in G.nodes()]}).sort_values(['weighted_degree','degree'],ascending=False)

def main():
    prepared={}; all_wdeg=[]; all_ew=[]
    print('='*72); print('THREE-PANEL STX SEMANTIC KNOWLEDGE GRAPH FIGURE'); print('='*72)
    for panel,cfg in NETWORKS.items():
        path=find_input_file(cfg['candidates'],panel)
        edges,node_types=load_and_standardize(path)
        filtered=filter_edges(edges)
        G=build_graph(filtered,node_types)
        prepared[panel]={'config':cfg,'input':path,'all_edges':edges,'filtered':filtered,'node_types':node_types,'graph':G}
        all_wdeg.extend(float(v) for _,v in G.degree(weight='weight'))
        all_ew.extend(float(d.get('weight',1)) for *_,d in G.edges(data=True))
        print(f"Panel {panel}: {G.number_of_nodes()} nodes, {G.number_of_edges()} plotted edges | {path}")
    gdmin,gdmax=min(all_wdeg),max(all_wdeg); ewmin,ewmax=min(all_ew),max(all_ew)
    fig=plt.figure(figsize=FIGSIZE)
    gs=fig.add_gridspec(2,2,hspace=.18,wspace=.10)
    axes={'A':fig.add_subplot(gs[0,0]),'B':fig.add_subplot(gs[0,1]),'C':fig.add_subplot(gs[1,:])}
    stats_all=[]
    for panel in ['A','B','C']:
        item=prepared[panel]; cfg=item['config']
        stats=draw_network(axes[panel],item['graph'],cfg['title'],panel,gdmin,gdmax,ewmin,ewmax)
        stats.insert(0,'panel',panel); stats.insert(1,'network',cfg['output_stem']); stats_all.append(stats)
        item['filtered'].to_csv(OUT_DIR/f"{cfg['output_stem']}_plotted_edges.csv",index=False)
        stats.to_csv(OUT_DIR/f"{cfg['output_stem']}_node_statistics.csv",index=False)
    handles=[Line2D([0],[0],marker='o',linestyle='None',markerfacecolor=ENTITY_COLORS[e],markeredgecolor='white',markersize=9,label=ENTITY_LABELS[e]) for e in ['TOXIN','SXT_GENE','DINO_TAXON','CYANO_TAXON','ENV_FACTOR','BIOLOGICAL_PROCESS','DETECTION_METHOD']]
    fig.legend(handles=handles,title='Entity type',loc='lower center',bbox_to_anchor=(.5,.012),ncol=4,frameon=False,fontsize=LEGEND_FONT_SIZE,title_fontsize=LEGEND_FONT_SIZE+1)
    fig.suptitle('Temporal and Cross-Taxa Semantic Knowledge Graphs of Saxitoxin Research',fontsize=21,fontweight='bold',y=.995)
    plt.subplots_adjust(top=.95,bottom=.085,left=.025,right=.985)
    fig.savefig(OUT_PNG,dpi=PNG_DPI,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT_PDF,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT_SVG,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.concat(stats_all,ignore_index=True).to_csv(OUT_DIR/'Figure_three_STX_semantic_KGs_node_statistics.csv',index=False)
    print('\nSaved:'); print(OUT_PNG); print(OUT_PDF); print(OUT_SVG)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
