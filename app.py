from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import sqlite3, secrets, io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = FastAPI(
    title="DATA CRISIS 2026 — Nova Aurora",
    version="2.0.0",
    description="Laboratório progressivo de Data Science em uma cidade inteligente fictícia."
)

DB = "lab.db"
rng = np.random.default_rng(20261005)

# -------------------- BANCO DE EQUIPES --------------------
def db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS equipes(
            token TEXT PRIMARY KEY,
            nome TEXT NOT NULL,
            fase INTEGER NOT NULL DEFAULT 1,
            tentativas_final INTEGER NOT NULL DEFAULT 0
        )
    """)
    con.commit()
    return con

def auth(token):
    if not token:
        raise HTTPException(401, "Informe o token da equipe no cabeçalho X-Team-Token.")
    con = db()
    row = con.execute("SELECT token,nome,fase,tentativas_final FROM equipes WHERE token=?", (token,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(401, "Token inválido.")
    return {"token": row[0], "nome": row[1], "fase": row[2], "tentativas_final": row[3]}

def set_fase(token, fase):
    con = db()
    con.execute("UPDATE equipes SET fase=? WHERE token=?", (fase, token))
    con.commit()
    con.close()

def require(team, fase):
    if team["fase"] < fase:
        raise HTTPException(403, f"Fase {fase} bloqueada. Conclua a fase anterior.")

# -------------------- DADOS SINTÉTICOS --------------------
SETORES = np.array(["SAUDE","ENERGIA","AGUA","TRANSITO","TELECOM"])

def gerar_base(n, modo):
    probs = [0.24,0.25,0.17,0.19,0.15] if modo=="historico" else [0.12,0.17,0.14,0.31,0.26]
    setor = rng.choice(SETORES, n, p=probs)
    idade = rng.integers(2,145,n)
    carga = np.clip(rng.normal(62 if modo=="historico" else 67,16,n),8,100)
    temp = np.array([rng.normal({"SAUDE":36,"ENERGIA":43,"AGUA":34,"TRANSITO":39,"TELECOM":42}[s]+0.08*c,4.5) for s,c in zip(setor,carga)])
    vib = np.clip(rng.normal(1.2,.45,n)+.017*carga+.35*(setor=="ENERGIA")+.25*(setor=="AGUA"),0,None)
    consumo = np.clip(25+1.45*carga+.18*idade+rng.normal(0,14,n),5,None)
    lat = np.clip(rng.gamma(2.1,24,n)+18*(setor=="TELECOM"),2,350)
    erros = rng.poisson(np.clip(1.2+carga/35,.5,7))
    manut = rng.poisson(.6,n)
    umid = np.clip(rng.normal(62,15,n),15,100)
    z = (-4.1+.026*(temp-38)+.42*(vib-1.8)+.022*(carga-60)+.14*erros
         +.004*(lat-45)+.004*(idade-55)+.22*(setor=="ENERGIA")+.18*(setor=="TELECOM")-.12*manut)
    z += .35*((temp>52)&(vib>3))
    z += .30*((carga>82)&(erros>=5))
    p = 1/(1+np.exp(-z))
    falha = rng.binomial(1,p)
    prefix = "H" if modo=="historico" else ("N" if modo=="novo" else "OP")
    df = pd.DataFrame({
        "unidade_id":[f"{prefix}{i:05d}" for i in range(1,n+1)],
        "setor":setor,
        "temperatura":np.round(temp,2),
        "vibracao":np.round(vib,3),
        "consumo_energia":np.round(consumo,2),
        "latencia_rede":np.round(lat,2),
        "carga_sistema":np.round(carga,2),
        "erros_24h":erros,
        "manutencoes_30d":manut,
        "idade_equipamento_meses":idade,
        "umidade":np.round(umid,2),
        "falha":falha
    })
    for col,rate in {"vibracao":.035,"latencia_rede":.025,"umidade":.02}.items():
        df.loc[rng.random(n)<rate,col] = np.nan
    if modo=="historico":
        dup = df.sample(120, random_state=11)
        df = pd.concat([df,dup],ignore_index=True)
    elif modo=="novo":
        idx = rng.choice(df.index,size=int(.18*n),replace=False)
        df.loc[idx,"carga_sistema"] = np.round(df.loc[idx,"carga_sistema"]/100,4)
    return df.sample(frac=1,random_state=22).reset_index(drop=True)

HIST = gerar_base(8000,"historico")
NOVOS = gerar_base(1500,"novo")
FINAL = gerar_base(700,"final")
# final sem problema de escala
FINAL.loc[FINAL["carga_sistema"]<=1,"carga_sistema"] *= 100

DUP_COUNT = int(HIST.duplicated().sum())
MOST_MISSING = str(HIST.isna().sum().drop("falha").idxmax())
p_hist = HIST.drop_duplicates()["setor"].value_counts(normalize=True)
p_new = NOVOS["setor"].value_counts(normalize=True)
SETOR_AUMENTO = str((p_new-p_hist).sort_values(ascending=False).index[0])

# -------------------- MODELOS Pydantic --------------------
class NovaEquipe(BaseModel):
    nome: str = Field(min_length=2,max_length=80)

class Resp1(BaseModel):
    duplicatas: int
    coluna_mais_ausente: str

class Resp2(BaseModel):
    setor_maior_aumento: str

class Resp3(BaseModel):
    coluna_descartar: str

class Resp4(BaseModel):
    coluna_alterada: str
    fator_correcao: float
    percentual_afetado_aprox: float | None = None

# -------------------- VISUAL EMBUTIDO --------------------
CITY_SVG = r"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#071426"/><stop offset=".58" stop-color="#163b59"/><stop offset="1" stop-color="#e7a05c"/></linearGradient>
<filter id="g"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect width="1600" height="900" fill="url(#sky)"/>
<path d="M0 500 Q220 330 430 470 T820 430 T1180 460 T1600 400 V900 H0Z" fill="#102b3b" opacity=".86"/>
<rect y="540" width="1600" height="360" fill="#0b1a24"/>
<g fill="#10283a" stroke="#294559" stroke-width="3">
<rect x="90" y="555" width="150" height="210"/><rect x="280" y="620" width="120" height="145"/><rect x="435" y="505" width="190" height="260"/>
<rect x="665" y="590" width="150" height="175"/><rect x="855" y="480" width="165" height="285"/><rect x="1055" y="575" width="120" height="190"/>
<rect x="1200" y="525" width="180" height="240"/><rect x="1410" y="615" width="110" height="150"/>
</g>
<path d="M0 815 C340 700 630 790 910 690 S1300 640 1600 760" fill="none" stroke="#ffd27a" stroke-width="5" opacity=".9"/>
<g stroke="#5ec7ff" fill="none" stroke-width="3" opacity=".7">
<path d="M185 530 C350 420 510 430 560 500"/><path d="M560 500 C770 380 930 390 940 470"/><path d="M940 470 C1130 410 1280 440 1285 520"/>
</g>
<g fill="#8be0ff" filter="url(#g)"><circle cx="185" cy="530" r="8"/><circle cx="560" cy="500" r="8"/><circle cx="940" cy="470" r="8"/><circle cx="1285" cy="520" r="8"/></g>
<g transform="translate(690 590)"><rect width="130" height="100" rx="8" fill="#dceaf4" opacity=".9"/><rect x="55" y="-28" width="20" height="55" fill="#dceaf4"/><rect x="38" y="-10" width="55" height="20" fill="#dceaf4"/></g>
<text x="90" y="120" font-family="Arial" font-size="58" font-weight="700" fill="#fff">NOVA AURORA</text>
<text x="94" y="164" font-family="Arial" font-size="22" fill="#bcd6e8" letter-spacing="5">CIDADE CONECTADA • DADOS EM TEMPO REAL</text>
</svg>
"""

CSS = r"""
:root{--bg:#07111f;--panel:#0f1d2f;--line:#263c55;--text:#f4f7fb;--muted:#a9b8c8;--accent:#62b7ff;--ok:#69e3c1;--warn:#ffca6a;--danger:#ff7a85}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.55}a{color:inherit}
.hero{min-height:72vh;display:flex;align-items:flex-end;position:relative;overflow:hidden;background:linear-gradient(180deg,rgba(3,10,19,.04),rgba(3,10,19,.9)),url('/cidade.svg') center/cover no-repeat;border-bottom:1px solid var(--line)}
.hero-inner,.container{width:min(1180px,92%);margin:auto}.hero-inner{padding:90px 0 54px}.kicker{text-transform:uppercase;letter-spacing:.18em;font-size:.8rem;color:#9dd4ff;font-weight:800}
.hero h1{font-size:clamp(3rem,8vw,6.6rem);line-height:.95;margin:.25rem 0}.hero p{max-width:760px;color:#d8e4f0;font-size:1.18rem}
.container{padding:44px 0 70px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}
.card{background:linear-gradient(180deg,#14263b,#0d1b2c);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:0 18px 45px rgba(0,0,0,.18)}.body{padding:20px}.card p{color:var(--muted)}
.button{display:inline-block;text-decoration:none;border:0;border-radius:11px;padding:12px 16px;font-weight:800;background:var(--accent);color:#06111d;cursor:pointer}.button.ghost{background:transparent;color:#fff;border:1px solid var(--line)}
.status{display:flex;gap:8px;flex-wrap:wrap}.badge{padding:7px 10px;border-radius:999px;background:#172a40;border:1px solid #2f4864;font-size:.82rem}.badge.ok{border-color:#2d6d4a;color:#bdf4cd}
.phase.locked{opacity:.48;filter:saturate(.6)}.phase .number{color:#82c8ff;font-weight:900;font-size:.75rem;letter-spacing:.13em}
.city-map{position:relative;min-height:500px;border:1px solid var(--line);border-radius:20px;overflow:hidden;background:linear-gradient(180deg,rgba(3,10,17,.08),rgba(3,10,17,.55)),url('/cidade.svg') center/cover no-repeat}
.hotspot{position:absolute;z-index:3;transform:translate(-50%,-50%)}.hotspot button{border:0;width:18px;height:18px;border-radius:50%;background:#77d8ff;box-shadow:0 0 0 0 rgba(119,216,255,.65);animation:pulse 2.2s infinite}.hotspot.warn button{background:var(--warn)}.hotspot.alert button{background:var(--danger)}
.hotspot span{position:absolute;left:25px;top:-9px;white-space:nowrap;padding:7px 9px;border-radius:8px;background:rgba(4,13,22,.86);border:1px solid #2d4963;font-size:.78rem}
@keyframes pulse{0%{transform:scale(.8);box-shadow:0 0 0 0 rgba(119,216,255,.6)}60%{transform:scale(1);box-shadow:0 0 0 16px rgba(119,216,255,0)}100%{transform:scale(.8)}}
.sensor{position:absolute;width:14px;height:14px;border-radius:50%;background:#7be6ff;animation:pulse 2.3s infinite}
.live{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.live span{padding:8px 11px;border-radius:999px;background:rgba(8,22,35,.72);border:1px solid rgba(121,180,224,.35);font-size:.84rem}
input{width:min(520px,100%);padding:13px;border-radius:10px;border:1px solid #36506b;background:#081421;color:white}
.console{background:#030a12;border:1px solid #223448;border-radius:12px;padding:18px;color:#c9e6ff;font-family:monospace}
.section-title{margin:42px 0 18px}.section-title h2{margin-bottom:.15rem}.section-title p{color:var(--muted);margin-top:0}
footer{color:var(--muted);border-top:1px solid var(--line);margin-top:58px;padding:28px 0}
"""

INDEX = r"""
<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DATA CRISIS 2026</title><link rel="stylesheet" href="/style.css"></head>
<body>
<section class="hero">
<span class="sensor" style="left:18%;top:62%"></span><span class="sensor" style="left:38%;top:55%;animation-delay:.7s"></span><span class="sensor" style="left:62%;top:56%;animation-delay:1.1s"></span><span class="sensor" style="left:82%;top:52%;animation-delay:.3s"></span>
<div class="hero-inner"><div class="kicker">Hackathon • Laboratório de Data Science</div><h1>Nova Aurora</h1>
<p>Uma cidade conectada. Milhares de sensores. Falhas inesperadas. Sua equipe foi convocada para transformar dados operacionais em decisões.</p>
<a class="button" href="#missao">Entrar na Central</a>
<div class="live"><span>Sensores ativos: <strong id="sensors">2.846</strong></span><span>Pacotes/min: <strong id="packets">18.420</strong></span><span>Status: monitoramento contínuo</span></div>
</div></section>
<main class="container">
<div class="section-title" id="missao"><h2>Operação Sinal Fraco</h2><p>A Central Integrada de Operações detectou ocorrências que não podem ser explicadas apenas por alarmes convencionais.</p></div>
<section class="grid">
<div class="card"><div class="body"><h3>Saúde</h3><p>Hospitais, UPAs e equipamentos críticos monitorados.</p></div></div>
<div class="card"><div class="body"><h3>Energia</h3><p>Subestações e pontos de distribuição sob diferentes cargas.</p></div></div>
<div class="card"><div class="body"><h3>Água</h3><p>Estações de bombeamento e reservatórios instrumentados.</p></div></div>
<div class="card"><div class="body"><h3>Trânsito</h3><p>Semáforos e controladores viários enviando dados continuamente.</p></div></div>
<div class="card"><div class="body"><h3>Telecom</h3><p>A rede conecta sensores, unidades e a Central.</p></div></div>
</section>
<div class="section-title"><h2>Iniciar operação</h2><p>Crie sua equipe para gerar o token de acesso.</p></div>
<section class="card"><div class="body"><input id="name" placeholder="Ex.: Equipe Turing"> <button class="button" onclick="createTeam()">Criar equipe</button><p id="result"></p></div></section>
<footer>DATA CRISIS 2026 • Nova Aurora é uma cidade fictícia criada para fins educacionais.</footer>
</main>
<script>
setInterval(()=>{let a=document.getElementById('sensors'),b=document.getElementById('packets');if(a)a.textContent=(2846+Math.floor(Math.random()*9-4)).toLocaleString('pt-BR');if(b)b.textContent=(18420+Math.floor(Math.random()*420-210)).toLocaleString('pt-BR')},2400);
async function createTeam(){let nome=document.getElementById('name').value.trim();if(!nome)return;let r=await fetch('/equipes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nome})});let d=await r.json();localStorage.setItem('nova_token',d.token);localStorage.setItem('nova_nome',nome);document.getElementById('result').innerHTML='Token criado: <strong>'+d.token+'</strong>';setTimeout(()=>location.href='/laboratorio',600)}
</script></body></html>
"""

LAB = r"""
<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Laboratório — Nova Aurora</title><link rel="stylesheet" href="/style.css"></head>
<body><section style="background:linear-gradient(135deg,#0d1e31,#15334f);border-bottom:1px solid #263c55"><div class="container" style="padding-top:35px;padding-bottom:35px"><div class="kicker">Central Integrada de Operações</div><h1>Painel da Equipe</h1><div class="status"><span class="badge ok">Equipe: <b id="team">—</b></span><span class="badge">Fase: <b id="phase">—</b></span><a class="badge" href="/docs" target="_blank">API /docs</a></div></div></section>
<main class="container" id="root">
<div class="section-title"><h2>Mapa operacional</h2><p>Os indicadores mudam conforme a investigação avança.</p></div>
<section class="city-map">
<div class="hotspot" id="h1" style="left:45%;top:69%"><button></button><span>Saúde</span></div>
<div class="hotspot" id="h2" style="left:69%;top:54%"><button></button><span>Energia</span></div>
<div class="hotspot" id="h3" style="left:28%;top:66%"><button></button><span>Água</span></div>
<div class="hotspot" id="h4" style="left:59%;top:78%"><button></button><span>Trânsito</span></div>
<div class="hotspot" id="h5" style="left:78%;top:43%"><button></button><span>Telecom</span></div>
</section>
<div class="section-title"><h2>Fases da investigação</h2><p>Conclua cada checkpoint na API para desbloquear a próxima fase.</p></div>
<section class="grid">
<div class="card phase" id="p1"><div class="body"><div class="number">FASE 01</div><h3>Diagnóstico inicial</h3><p>Conheça a base histórica.</p><button class="button" onclick="openPhase(1)">Abrir</button></div></div>
<div class="card phase locked" id="p2"><div class="body"><div class="number">FASE 02</div><h3>Nova remessa</h3><p>Compare dados novos ao histórico.</p><button class="button" onclick="openPhase(2)">Abrir</button></div></div>
<div class="card phase locked" id="p3"><div class="body"><div class="number">FASE 03</div><h3>Dado comprometido</h3><p>Revise o pipeline.</p><button class="button" onclick="openPhase(3)">Abrir</button></div></div>
<div class="card phase locked" id="p4"><div class="body"><div class="number">FASE 04</div><h3>Inconsistência oculta</h3><p>Descubra o problema de escala.</p><button class="button" onclick="openPhase(4)">Abrir</button></div></div>
<div class="card phase locked" id="p5"><div class="body"><div class="number">FASE 05</div><h3>Operação real</h3><p>Submeta o modelo final.</p><button class="button" onclick="openPhase(5)">Abrir</button></div></div>
</section>
<div class="section-title"><h2>Terminal da missão</h2></div><div id="detail" class="card"><div class="body"><p>Selecione uma fase.</p></div></div>
<footer>Nova Aurora • Laboratório progressivo de Data Science</footer>
</main>
<script>
const token=()=>localStorage.getItem('nova_token');
async function load(){if(!token()){document.getElementById('root').innerHTML='<p>Volte à página inicial e crie sua equipe.</p>';return}
let r=await fetch('/progresso',{headers:{'X-Team-Token':token()}}),d=await r.json();if(!r.ok)return;
document.getElementById('team').textContent=d.equipe;document.getElementById('phase').textContent=d.fase_atual;
for(let i=1;i<=5;i++){let e=document.getElementById('p'+i);if(i<=d.fase_atual)e.classList.remove('locked')}
['h1','h2','h3','h4','h5'].forEach(x=>document.getElementById(x).classList.remove('warn','alert'));
if(d.fase_atual==2){document.getElementById('h4').classList.add('warn');document.getElementById('h5').classList.add('warn')}
if(d.fase_atual==3){document.getElementById('h1').classList.add('warn');document.getElementById('h2').classList.add('warn')}
if(d.fase_atual>=4){document.getElementById('h2').classList.add('alert');document.getElementById('h4').classList.add('alert');document.getElementById('h5').classList.add('alert')}
}
async function openPhase(n){let r=await fetch('/fase/'+n,{headers:{'X-Team-Token':token()}}),d=await r.json();let out=document.getElementById('detail');if(!r.ok){out.innerHTML='<div class="body"><pre class="console">'+JSON.stringify(d,null,2)+'</pre></div>';return}
let li=(d.missao||[]).map(x=>'<li>'+x+'</li>').join('');out.innerHTML='<div class="body"><div class="number">FASE '+d.fase+'</div><h2>'+d.titulo+'</h2><p>'+(d.comunicado||d.cenario||'')+'</p><ul>'+li+'</ul>'+(d.arquivo?'<button class="button" onclick="downloadFile(\\''+d.arquivo+'\\')">Baixar dados</button> ':'')+'<a class="button ghost" target="_blank" href="/docs">Abrir /docs</a><p>'+((d.checkpoint||d.submissao)||'')+'</p></div>';out.scrollIntoView({behavior:'smooth'})}
async function downloadFile(url){let r=await fetch(url,{headers:{'X-Team-Token':token()}});let b=await r.blob();let a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=url.split('/').pop();a.click()}
load();setInterval(load,12000);
</script></body></html>
"""

@app.get("/style.css")
def style():
    return Response(CSS, media_type="text/css")

@app.get("/cidade.svg")
def cidade():
    return Response(CITY_SVG, media_type="image/svg+xml")

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX)

@app.get("/laboratorio", response_class=HTMLResponse)
def laboratorio():
    return HTMLResponse(LAB)

# -------------------- EQUIPES / PROGRESSO --------------------
@app.post("/equipes")
def criar_equipe(payload: NovaEquipe):
    token = secrets.token_hex(8)
    con = db()
    con.execute("INSERT INTO equipes(token,nome,fase,tentativas_final) VALUES(?,?,1,0)",(token,payload.nome))
    con.commit(); con.close()
    return {"equipe":payload.nome,"token":token,"fase_atual":1}

@app.get("/progresso")
def progresso(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token)
    return {"equipe":t["nome"],"fase_atual":t["fase"],"fases_concluidas":list(range(1,t["fase"])),"tentativas_final":t["tentativas_final"]}

# -------------------- DOWNLOAD CSV --------------------
def csv_response(df, filename):
    buf=io.StringIO()
    df.to_csv(buf,index=False)
    return Response(buf.getvalue(),media_type="text/csv",headers={"Content-Disposition":f'attachment; filename="{filename}"'})

@app.get("/dados/iniciais.csv")
def dados1(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,1)
    return csv_response(HIST,"dados_iniciais.csv")

@app.get("/dados/novos.csv")
def dados2(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,2)
    return csv_response(NOVOS,"novos_dados.csv")

@app.get("/dados/operacao-final.csv")
def dadosfinal(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,5)
    return csv_response(FINAL.drop(columns=["falha"]),"operacao_final.csv")

# -------------------- FASES --------------------
@app.get("/fase/1")
def fase1(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,1)
    return {"fase":1,"titulo":"Diagnóstico inicial","cenario":"A Central entrega a base histórica de Nova Aurora.","arquivo":"/dados/iniciais.csv","missao":["Inspecione estrutura e tipos.","Identifique duplicatas exatas.","Descubra qual coluna possui mais valores ausentes.","Faça análise exploratória antes da modelagem."],"checkpoint":"POST /fase/1/checkpoint"}

@app.post("/fase/1/checkpoint")
def cp1(payload: Resp1, x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,1)
    ok1=payload.duplicatas==DUP_COUNT
    ok2=payload.coluna_mais_ausente.strip().lower()==MOST_MISSING.lower()
    if not(ok1 and ok2):
        return {"aprovado":False,"feedback":{"duplicatas":"correto" if ok1 else "revise duplicatas exatas","ausentes":"correto" if ok2 else "compare NaN por coluna"}}
    set_fase(t["token"],2)
    return {"aprovado":True,"mensagem":"Fase 2 liberada.","proximo":"/fase/2"}

@app.get("/fase/2")
def fase2(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,2)
    return {"fase":2,"titulo":"Nova remessa","comunicado":"Uma nova remessa de dados operacionais foi recebida. Os novos registros devem ser analisados antes de serem incorporados.","arquivo":"/dados/novos.csv","missao":["Compare a composição da nova remessa com o histórico.","Não concatene automaticamente.","Descubra qual setor teve o maior aumento proporcional."],"checkpoint":"POST /fase/2/checkpoint"}

@app.post("/fase/2/checkpoint")
def cp2(payload: Resp2, x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,2)
    if payload.setor_maior_aumento.strip().upper()!=SETOR_AUMENTO.upper():
        return {"aprovado":False,"feedback":"Compare proporções por setor, não apenas contagens."}
    set_fase(t["token"],3)
    return {"aprovado":True,"mensagem":"Fase 3 liberada.","proximo":"/fase/3"}

@app.get("/fase/3")
def fase3(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,3)
    return {"fase":3,"titulo":"Variável comprometida","comunicado":"Foi confirmado um problema na coleta da variável temperatura. Ela não pode mais ser considerada confiável.","missao":["Retire a variável comprometida do pipeline.","Reavalie o modelo.","Compare o comportamento antes e depois."],"checkpoint":"POST /fase/3/checkpoint"}

@app.post("/fase/3/checkpoint")
def cp3(payload: Resp3, x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,3)
    if payload.coluna_descartar.strip().lower()!="temperatura":
        return {"aprovado":False,"feedback":"Releia o comunicado técnico."}
    set_fase(t["token"],4)
    return {"aprovado":True,"mensagem":"Fase 4 liberada.","proximo":"/fase/4"}

@app.get("/fase/4")
def fase4(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,4)
    return {"fase":4,"titulo":"Inconsistência oculta","comunicado":"Parte dos controladores foi substituída. Há indícios de que uma grandeza passou a ser transmitida em escala diferente do padrão histórico.","missao":["Compare distribuições numéricas.","Identifique a variável com duas escalas incompatíveis.","Estime a fração afetada.","Proponha a correção de escala."],"checkpoint":"POST /fase/4/checkpoint"}

@app.post("/fase/4/checkpoint")
def cp4(payload: Resp4, x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,4)
    okc=payload.coluna_alterada.strip().lower()=="carga_sistema"
    okf=90<=payload.fator_correcao<=110
    okp=True
    if payload.percentual_afetado_aprox is not None:
        p=payload.percentual_afetado_aprox
        if p<=1: p*=100
        okp=12<=p<=24
    if not(okc and okf and okp):
        return {"aprovado":False,"feedback":"Compare mínimos, quartis e histogramas entre histórico e nova remessa; procure uma diferença simples de escala."}
    set_fase(t["token"],5)
    return {"aprovado":True,"mensagem":"Desafio final liberado.","proximo":"/fase/5"}

@app.get("/fase/5")
def fase5(x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,5)
    return {"fase":5,"titulo":"Operação real","arquivo":"/dados/operacao-final.csv","missao":["Treine sua solução final com dados confiáveis.","Gere CSV com unidade_id e classe_prevista.","Você tem no máximo 3 submissões."],"submissao":"POST /final/submeter"}

@app.post("/final/submeter")
async def submeter(arquivo: UploadFile=File(...), x_team_token: str | None = Header(default=None)):
    t=auth(x_team_token); require(t,5)
    if t["tentativas_final"]>=3:
        raise HTTPException(403,"Limite de 3 submissões atingido.")
    pred=pd.read_csv(arquivo.file)
    if not {"unidade_id","classe_prevista"}.issubset(pred.columns):
        raise HTTPException(400,"CSV deve conter unidade_id e classe_prevista.")
    gab=FINAL[["unidade_id","falha"]]
    m=gab.merge(pred[["unidade_id","classe_prevista"]],on="unidade_id",how="left")
    if m["classe_prevista"].isna().any():
        raise HTTPException(400,"Existem unidades sem previsão.")
    y=m["falha"].astype(int); p=m["classe_prevista"].astype(int)
    con=db(); con.execute("UPDATE equipes SET tentativas_final=tentativas_final+1 WHERE token=?",(t["token"],)); con.commit(); con.close()
    return {"submissao":t["tentativas_final"]+1,"accuracy":round(float(accuracy_score(y,p)),4),"precision":round(float(precision_score(y,p,zero_division=0)),4),"recall":round(float(recall_score(y,p,zero_division=0)),4),"f1":round(float(f1_score(y,p,zero_division=0)),4)}
