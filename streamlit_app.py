import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="スマートコンソーシアム・シミュレーター & Graph Strategy", layout="wide")

# --- 2. セッション状態の初期化 ---
if 'master_db' not in st.session_state:
    st.session_state.master_db = pd.DataFrame()
if 'history_pts' not in st.session_state:
    st.session_state.history_pts = []
if 'graph' not in st.session_state:
    st.session_state.graph = nx.DiGraph()
if 'source_node' not in st.session_state:
    st.session_state.source_node = None
if 'target_node' not in st.session_state:
    st.session_state.target_node = None

# --- 3. 補助関数 ---
def find_pareto_front(costs, benefits):
    costs, benefits = np.array(costs), np.array(benefits)
    indices = np.arange(len(costs))
    pareto_front = []
    for i in indices:
        is_dominated = False
        for j in indices:
            if (costs[j] <= costs[i] and benefits[j] >= benefits[i]) and \
               (costs[j] < costs[i] or benefits[j] > benefits[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(i)
    return sorted(pareto_front, key=lambda x: costs[x])

def calculate_gini(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1 or np.sum(x) == 0: return 0
    n = len(x)
    diff_sum = np.sum(np.abs(x[:, None] - x))
    return diff_sum / (2 * n * np.sum(x))

# --- 4. 計算エンジン ---
def run_strategic_simulation(params, base_df):
    np.random.seed(42)
    UNIT_APC_INDIV = params['list_apc_price'] / 10000 
    
    if base_df.empty:
        sub_scale = 3.5 if params['pub_type'] == "Elsevier" else 1.2
        raw_list = []
        configs = [('Tier1', 30, sub_scale, 150), ('Tier2', 120, sub_scale*0.12, 30), ('Tier3', 50, sub_scale*0.02, 5)]
        for t_name, count, s_val, p_val in configs:
            for i in range(count):
                p_count = max(1, int(np.random.normal(p_val, p_val*0.25)))
                raw_list.append({
                    'Entity': f"{t_name}_{i}", 'Tier': t_name, 
                    'Access': float(max(5, int(np.random.normal(p_val*10, p_val)))),
                    'Total_Pubs': float(p_count),
                    'Base_Sub': float(max(s_val*0.6, np.random.normal(s_val, s_val*0.1))),
                    'Tokens': float(int(p_val * 1.1) if t_name == 'Tier1' else 0)
                })
        working_df = pd.DataFrame(raw_list)
    else:
        working_df = base_df.copy()

    for col in ['Access', 'Total_Pubs', 'Base_Sub', 'Tokens']:
        if col in working_df.columns:
            working_df[col] = pd.to_numeric(working_df[col], errors='coerce').fillna(0).astype(float)

    green_r = params['green_oa_rate'] / 100
    unbundle_r = params['unbundle_rate']
    
    noise = np.random.uniform(-0.1, 0.1, len(working_df))
    working_df['Green_OA_Pubs'] = working_df['Total_Pubs'] * green_r
    working_df['Gold_OA_Pubs'] = (working_df['Total_Pubs'] * (0.6 - green_r + noise)).clip(lower=0)
    working_df['Total_OA_Pubs'] = working_df['Green_OA_Pubs'] + working_df['Gold_OA_Pubs']
    
    negotiated_apc = UNIT_APC_INDIV * (params['target_apc_price'] / params['list_apc_price'])
    total_cons_sub = working_df['Base_Sub'].sum() * (1 - unbundle_r)
    total_cons_apc = working_df['Gold_OA_Pubs'].sum() * negotiated_apc
    
    actual_reqs = (working_df['Access'].sum() * unbundle_r * 0.05) * (1 - params['backfile_rate'])
    total_ill_cost = (actual_reqs * params['ill_cover_rate'] * 500) / 100000000
    total_ppv_cost = (actual_reqs * (1 - params['ill_cover_rate']) * 4000) / 100000000
    total_fund_cost = params['fund_investment'] / 10
    
    total_cons_cost = total_cons_sub + total_cons_apc + total_ill_cost + total_ppv_cost + total_fund_cost
    
    acc_s = working_df['Access'] / working_df['Access'].sum() if working_df['Access'].sum() > 0 else 0
    pub_s = working_df['Total_Pubs'] / working_df['Total_Pubs'].sum() if working_df['Total_Pubs'].sum() > 0 else 0
    working_df['Cons_Cost'] = total_cons_cost * (params['read_weight'] * acc_s + (1-params['read_weight']) * pub_s)
    
    working_df['Win_Loss'] = working_df['Base_Sub'] + (working_df['Total_Pubs'] * 0.4 * UNIT_APC_INDIV) - working_df['Cons_Cost']
    working_df['ROI'] = working_df['Total_OA_Pubs'] / working_df['Cons_Cost'].replace(0, np.nan)
    working_df['OA_Rate'] = (working_df['Total_OA_Pubs'] / working_df['Total_Pubs'] * 100).clip(upper=100)
    
    return total_cons_cost, working_df['Total_OA_Pubs'].sum(), total_cons_sub, total_cons_apc, total_ill_cost, total_ppv_cost, total_fund_cost, working_df

# --- 5. UI設定（サイドバー） ---
st.sidebar.title("🛡️ パラメータ設定")
p_type = st.sidebar.selectbox("対象出版社", ["Elsevier", "Wiley/Springer"])
g_oa = st.sidebar.slider("グリーンOA率 (%)", 0, 50, 25)
unb = st.sidebar.slider("購読削減(Unbundle)率", 0.0, 1.0, 0.40)
w_read = st.sidebar.slider("按分重み (利用 1.0 ↔ 出版 0.0)", 0.0, 1.0, 0.5)

st.sidebar.divider()
st.sidebar.subheader("🏦 基金・ILL設定")
fund_inv = st.sidebar.number_input("基金投資額 (億円)", value=50)
bf_rate = st.sidebar.slider("バックファイル購入率 (%)", 0, 100, 40)
ill_r = st.sidebar.slider("スマートILLカバー率 (%)", 0, 100, 85)

st.sidebar.divider()
st.sidebar.subheader("📐 共通グラフスケール")
use_custom_scale = st.sidebar.checkbox("軸の範囲を手動設定する", value=False)
if p_type == "Elsevier": def_x, def_y = [50, 250], [3000, 12000]
else: def_x, def_y = [20, 120], [1000, 6000]
if use_custom_scale:
    shared_x_range = [st.sidebar.number_input("コスト最小", value=float(def_x[0])), st.sidebar.number_input("コスト最大", value=float(def_x[1]))]
    shared_y_range = [st.sidebar.number_input("OA数最小", value=float(def_y[0])), st.sidebar.number_input("OA数最大", value=float(def_y[1]))]
else:
    shared_x_range, shared_y_range = def_x, def_y

params = {'pub_type': p_type, 'green_oa_rate': g_oa, 'unbundle_rate': unb, 'read_weight': w_read, 'fund_investment': fund_inv, 'backfile_rate': bf_rate/100, 'indiv_burden_rate': 0.2, 'smart_ill_unit_cost': 500, 'ill_cover_rate': ill_r/100, 'req_rate': 0.05, 'ppv_unit_price': 4000, 'list_apc_price': 45, 'target_apc_price': 30}

if st.sidebar.button("履歴をリセット"):
    st.session_state.history_pts = []
    st.rerun()

# --- 6. 計算実行 ---
res = run_strategic_simulation(params, st.session_state.master_db)
total_cost, total_oa, sub_c, apc_c, ill_c, ppv_c, fund_c, df_final = res
st.session_state.history_pts.append({'cost': total_cost, 'oa': total_oa})

# --- 7. メイン画面 ---
st.title("スマートコンソーシアム・シミュレーター & Graph Strategy")

tabs = st.tabs(["📈 ダッシュボード", "💎 パレートグリッド探索", "⚖️ ティア別評価", "🔄 トークン融通(Sankey)", "🏫 OA分析", "🕸️ 関係性グラフ(得失サイズ)", "💾 データ管理"])

with tabs[0]: # ダッシュボード
    st.header("■戦略現状分析")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("法人総コスト", f"{total_cost:.1f} 億円")
    c2.metric("総OA論文数", f"{total_oa:.0f} 本")
    c3.metric("ILL想定件数", f"{(ill_c*100000000/500):,.0f} 件")
    c4.metric("インフラ費用(年)", f"{fund_c:.1f} 億円")
    st.divider()
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("🎯 パレート・フロンティアと履歴")
        hist_df = pd.DataFrame(st.session_state.history_pts)
        c_theory = np.linspace(shared_x_range[0], shared_x_range[1], 100)
        oa_theory = total_oa * (c_theory / total_cost)**0.65
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_theory, y=oa_theory, mode='lines', name='理論限界', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=hist_df['cost'], y=hist_df['oa'], mode='markers', name='履歴', marker=dict(color='gray', opacity=0.3)))
        fig.add_trace(go.Scatter(x=[total_cost], y=[total_oa], mode='markers+text', text=["現在"], marker=dict(color='blue', size=20, symbol='star')))
        fig.update_layout(xaxis=dict(range=shared_x_range, title="コスト(億円)"), yaxis=dict(range=shared_y_range, title="OA数"), height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("💰 予算構造")
        fig_pie = go.Figure(data=[go.Pie(labels=['購読費', 'APC', 'ILL', 'PPV', '基金'], values=[sub_c, apc_c, ill_c, ppv_c, fund_c], hole=.4)])
        st.plotly_chart(fig_pie, use_container_width=True)

with tabs[1]: # パレート探索
    st.header("💎 網羅的パレート最適化探索 (Grid Search)")
    if st.button("全探索を実行 (400パターン)"):
        all_res = []
        g_space, u_space = np.linspace(0, 50, 20), np.linspace(0, 1.0, 20)
        prog = st.progress(0); count = 0
        for g in g_space:
            for u in u_space:
                tp = params.copy(); tp['green_oa_rate'], tp['unbundle_rate'] = g, u
                tc, toa, _, _, _, _, _, _ = run_strategic_simulation(tp, st.session_state.master_db)
                all_res.append({'cost': tc, 'oa': toa, 'green': g, 'unbundle': u})
                count += 1; prog.progress(count / 400)
        res_df = pd.DataFrame(all_res)
        p_idx = find_pareto_front(res_df['cost'].values, res_df['oa'].values)
        pareto_df = res_df.iloc[p_idx]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=res_df['cost'], y=res_df['oa'], mode='markers', name='全探索案', marker=dict(color='lightgray', opacity=0.5)))
        fig_p.add_trace(go.Scatter(x=pareto_df['cost'], y=pareto_df['oa'], mode='lines+markers', name='パレート最効率', line=dict(color='red', width=3)))
        fig_p.add_trace(go.Scatter(x=[total_cost], y=[total_oa], mode='markers', name='現在の設定', marker=dict(color='blue', size=15, symbol='star')))
        fig_p.update_layout(xaxis=dict(range=shared_x_range, title="コスト(億円)"), yaxis=dict(range=shared_y_range, title="OA数"), height=600, template="plotly_white")
        st.plotly_chart(fig_p, use_container_width=True)

with tabs[2]: # ティア別評価
    st.header("⚖️ ティア別得失と公平性評価")
    gini = calculate_gini(df_final['Win_Loss'])
    st.metric("得失不平等度 (ジニ係数)", f"{gini:.3f}")
    cl1, cl2 = st.columns(2)
    with cl1:
        fig_wl = px.box(df_final, x='Tier', y='Win_Loss', color='Tier', points="all", title="現状比削減額 (億円/校)")
        fig_wl.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_wl, use_container_width=True)
    with cl2:
        fig_roi = px.violin(df_final, x='Tier', y='ROI', box=True, color='Tier', title="投資効率 (OA数/億円)")
        st.plotly_chart(fig_roi, use_container_width=True)

with tabs[3]: # Sankey
    st.header("🔄 転換契約トークンの循環 (Sankey)")
    t1_excess = max(1, df_final[df_final['Tier']=='Tier1']['Tokens'].sum() - df_final[df_final['Tier']=='Tier1']['Gold_OA_Pubs'].sum())
    st.plotly_chart(go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, label=["Tier1余剰", "法人Pool", "Tier2需要", "Tier3需要"]), link=dict(source=[0, 1, 1], target=[1, 2, 3], value=[t1_excess, t1_excess*0.8, t1_excess*0.2]))]), use_container_width=True)

with tabs[4]: # OA分析
    st.header("■機関別:OA転換状況のプロファイリング分析")
    st.write("各大学の出版規模とOA転換率の相関を可視化します。特定の範囲を詳細に見るために縦軸を調整できます。")
    c_p1, c_p2 = st.columns(2)
    with c_p1: y_prof_range = st.slider("OA転換率の表示範囲 (%)", 0, 100, (40, 80))
    with c_p2: st.info("💡 ティアごとに近似曲線を表示しています（OLS回帰）。")
    
    fig_prof = px.scatter(df_final, x="Total_Pubs", y="OA_Rate", size="Total_OA_Pubs", color="Tier", 
                         hover_name="Entity", trendline="ols",
                         labels={"Total_Pubs": "年間総出版数", "OA_Rate": "OA転換率 (%)"})
    fig_prof.update_layout(height=550, yaxis=dict(range=[y_prof_range[0], y_prof_range[1]]), margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_prof, use_container_width=True)

with tabs[5]: # 関係性グラフ
    st.header("🕸️ コンソーシアム構造可視化 (得失サイズ反映)")
    col_g, col_ctrl = st.columns([3, 1])
    with col_g:
        nodes = []
        edges = []
        nodes.append(Node(id="Consortium_Fund", label="💰 基金", size=40, color="#FFD700"))
        nodes.append(Node(id="Publisher", label=f"🏢 {p_type}", size=40, color="#C0C0C0"))
        for _, row in df_final.iterrows():
            wl = row['Win_Loss']
            impact_size = 15 + (abs(wl) * 20)
            n_color = "#4CAF50" if wl > 0 else "#F44336"
            nodes.append(Node(id=row['Entity'], label=row['Entity'], size=impact_size, color=n_color))
            edges.append(Edge(source=row['Entity'], target="Publisher", label="Pay", color="#D3D3D3"))
        for u, v, d in st.session_state.graph.edges(data=True):
            edges.append(Edge(source=u, target=v, label=d.get('relation', 'link'), color="#5D5CDE"))
        config = Config(width="100%", height=600, directed=True, physics=True)
        sel_node = agraph(nodes=nodes, edges=edges, config=config)
    with col_ctrl:
        st.subheader("🛠️ ノード詳細と操作")
        if sel_node:
            st.success(f"選択中: **{sel_node}**")
            if sel_node in df_final['Entity'].values:
                node_data = df_final[df_final['Entity'] == sel_node].iloc[0]
                st.metric("得失額 (億円)", f"{node_data['Win_Loss']:.3f}")
                st.write(f"ティア: {node_data['Tier']}")
                st.write(f"出版数: {node_data['Total_Pubs']:.0f} 本")
                st.write(f"OA率: {node_data['OA_Rate']:.1f} %")
            st.divider()
            c1, c2 = st.columns(2)
            if c1.button("始点に設定"): st.session_state.source_node = sel_node
            if c2.button("終点に設定"): st.session_state.target_node = sel_node
        else:
            st.info("グラフ上のノードをクリックしてください")
        st.write(f"**From:** {st.session_state.source_node if st.session_state.source_node else '-'}")
        st.write(f"**To:** {st.session_state.target_node if st.session_state.target_node else '-'}")
        if st.session_state.source_node and st.session_state.target_node:
            rel = st.text_input("関係名を入力")
            if st.button("🔗 接続を定義"):
                st.session_state.graph.add_edge(st.session_state.source_node, st.session_state.target_node, relation=rel)
                st.session_state.source_node = st.session_state.target_node = None; st.rerun()
        if st.button("選択・グラフをリセット"): 
            st.session_state.graph.clear()
            st.session_state.source_node = st.session_state.target_node = None; st.rerun()

with tabs[6]: # データ管理
    st.header("💾 データ連携・管理")
    col_up, col_down = st.columns(2)
    with col_up:
        st.subheader("●データのインポート")
        f_m = st.file_uploader("1. 大学マスタCSV", type="csv")
        if f_m: st.session_state.master_db = pd.read_csv(f_m); st.rerun()
        f_e = st.file_uploader("2. EZproxyログCSV", type="csv")
        if f_e and not st.session_state.master_db.empty:
            log = pd.read_csv(f_e)
            upd = st.session_state.master_db.merge(log[['Entity', 'Log_Count']], on='Entity', how='left')
            upd['Access'] = upd['Log_Count'].fillna(upd['Access'])
            st.session_state.master_db = upd.drop(columns=['Log_Count']); st.rerun()
        
        # テンプレートダウンロード機能
        st.divider()
        st.subheader("📋 テンプレートの取得")
        st.write("新規データ作成用の空のテンプレートファイルをダウンロードします。")
        template_df = pd.DataFrame(columns=['Entity', 'Tier', 'Access', 'Total_Pubs', 'Base_Sub', 'Tokens'])
        csv_template = template_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📄 空のテンプレートをダウンロード",
            data=csv_template,
            file_name="university_master_template.csv",
            mime='text/csv'
        )

    with col_down:
        st.subheader("◆データのエクスポート")
        st.download_button("分析結果をCSVでダウンロード", df_final.to_csv(index=False).encode('utf-8-sig'), "simulation_result.csv")
        
        if not st.session_state.master_db.empty:
            template_cols = ['Entity', 'Tier', 'Access', 'Total_Pubs', 'Base_Sub', 'Tokens']
            current_master = df_final[template_cols].to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📁 現在のマスタ(入力形式)を保存", data=current_master, file_name="current_university_master.csv", mime='text/csv')
    
    st.divider()
    if not st.session_state.master_db.empty:
        st.subheader("●読み込み済みデータ (先頭5件)")
        st.dataframe(st.session_state.master_db.head())