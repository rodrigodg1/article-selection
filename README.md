# article-selection

## Para que serve este repositório

Este repositório contém uma **ferramenta local de triagem de literatura científica**. O objetivo é apoiar **revisões sistemáticas, mapeamentos bibliográficos ou qualquer processo em que uma pessoa precise decidir, registo a registo, se um artigo entra ou não num estudo** (inclusão / exclusão), com registo dessa decisão e possibilidade de rever mais tarde.

**Problema que resolve:** exportas da Web of Science, Scopus, PubMed, Lens, etc. um CSV com centenas ou milhares de referências; precisas de **ler título e resumo**, marcar **sim** ou **não**, eventualmente **anotar o motivo**, **agrupar por temas** (categorias) e **voltar atrás** sem perder o trabalho. Planilhas puras tornam isso lento e frágil; esta app dá um fluxo dedicado no browser.

**O que não é:** não executa buscas em bases externas, não deduplica automaticamente por DOI em massa e não substitui critérios de elegibilidade definidos no protocolo da revisão — **a decisão é sempre humana**; a app apenas organiza o CSV, a sessão de triagem e a exportação com rótulos e metadados.

---

## Como deve ser utilizado

Utiliza-se **no teu computador**, em **uma máquina de cada vez** (sessão Flask por browser). O fluxo típico é sempre o mesmo: **preparar CSV → subir → triar na interface → exportar CSV com decisões**.

### 1. Preparar o ficheiro de entrada

- Gera um **CSV** com os teus resultados de pesquisa (UTF-8 recomendado).
- Garante pelo menos uma coluna de **título** ou **resumo** (nomes reconhecidos estão na secção [Formato do CSV](#formato-do-csv)).
- Opcional: inclui colunas extra (DOI, revista, número de citações, etc.); ficam guardadas e saem de novo na exportação.

### 2. Instalar e arrancar a aplicação

Na pasta do repositório:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abre o browser em **http://127.0.0.1:5000/** (porta por omissão do Flask em modo desenvolvimento).

Na **primeira execução** é criada uma base **SQLite** local (por omissão `literature.db` no diretório de trabalho da app) com tabelas vazias.

### 3. Utilização no browser (fluxo diário)

| Passo | O que fazer |
|--------|----------------|
| **A. Carregar dados** | Vai a **Upload**, escolhe o CSV, opcionalmente preenche **nome do dataset** e **string de pesquisa** (só para documentares no export de onde vieram os registos). Submete. |
| **B. Triar** | A app leva-te ao ecrã de **rotulagem**: para cada artigo escolhe **Sim** ou **Não** (inclusão / exclusão), ou **Saltar** para voltar mais tarde; podes escrever **notas** (ex.: critério PICO, motivo de exclusão). |
| **C. Ajudar a leitura** | Na barra de navegação define **palavras-chave de destaque** (ou regex) para realçar termos no resumo e filtrar mentalmente o que é relevante. |
| **D. Rever em lote** | Usa **Gerir** para ver tabela com filtros (rótulo, categorias, “só com destaque no resumo”), **corrigir** decisões e **atribuir categorias** temáticas. |
| **E. Fechar o ciclo** | Usa **Exportar** para descarregar um **CSV com rótulos, notas, extras e categorias** — este ficheiro é o artefacto para o teu protocolo, PRISMA ou análise posterior em R/Python/Excel. |

**Vários conjuntos de resultados:** cada upload cria um **dataset**; na navegação podes **mudar o dataset ativo** para continuar outra ronda de triagem sem misturar ficheiros.

**Procura pontual:** a página **Busca por DOI** (`/search/doi?doi=…`) ajuda a localizar um registo no dataset ativo quando o identificador veio numa coluna extra.

### 4. Quando terminas ou mudas de máquina

- Guarda o **CSV exportado** (é a cópia de trabalho com decisões).
- Se quiseres backup completo, copia também o **ficheiro SQLite** da base de dados (contém todos os datasets e sessão de categorias).

### 5. Produção e segurança

- O modo `python app.py` é para **desenvolvimento / uso pessoal**.
- Em **produção** define `SECRET_KEY` forte e, se várias pessoas acederem, considera uma base partilhada (`SQLALCHEMY_DATABASE_URI`) e um servidor WSGI adequado — o modelo atual assume **uso local ou confiável**.

---

## Funcionalidades (resumo técnico)

- **Importação de CSV**: cria um *dataset*; colunas extras → `extra_json`; DOI / periódico / citações quando os cabeçalhos coincidem com o que a app reconhece.
- **Rotulagem sim/não** com notas e **Saltar**.
- **Categorias** (N:N): criar, atribuir, remover, renomear.
- **Destaques**: termos ou regex no resumo; página **Highlights** só com artigos que casam.
- **Gestão**: filtros por rótulo, categorias e destaque no resumo.
- **Exportação CSV** com metadados do dataset (`dataset_name`, `dataset_search_query`, etc.).

---

## Requisitos

- Python 3.9+.
- Dependências: `requirements.txt`.

---

## Variáveis de ambiente

| Variável | Descrição | Omissão |
|----------|-----------|---------|
| `SECRET_KEY` | Chave da sessão Flask (obrigatória em produção). | `dev-secret` |
| `SQLALCHEMY_DATABASE_URI` | URI SQLAlchemy (SQLite local, Turso `sqlite+libsql://…`, PostgreSQL, etc.). | `sqlite:///literature.db` |
| `TURSO_AUTH_TOKEN` | Token da base Turso (opcional). Usa-se com `SQLALCHEMY_DATABASE_URI` em formato `sqlite+libsql://…?secure=true` quando o token não vai no URL. | — |
| `SITE_ACCESS_PASSWORD` | Se definida, exige login por senha antes da app (útil em deploy público). | — |
| `USER_TIMEZONE` | Fuso para datas na interface (`local_dt`). | `America/Los_Angeles` |

### Vercel: não perder os dados do `literature.db` local

No plano **Hobby** da Vercel, o ficheiro SQLite em `/tmp` **não é fiável** (novo deploy ou outro *cold start* pode voltar a uma base vazia). Para **continuar com os mesmos artigos já rotulados**:

1. Cria uma base gratuita em **[Turso](https://turso.tech)** (SQLite na nuvem, compatível com o teu ficheiro).
2. Instala o [CLI Turso](https://docs.turso.tech/cli/overview), autentica-te e importa o ficheiro local (ajusta o caminho; por exemplo `./literature.db` ou `./instance/literature.db`):

   `turso db create litreview`  
   `turso db import litreview ./literature.db`

3. Obtém o URL da base (hostname `*.turso.io`) e um **token** com permissão de leitura/escrita.
4. No projeto Vercel → **Settings → Environment Variables**, define por exemplo:
   - `SQLALCHEMY_DATABASE_URI` = `sqlite+libsql://SEU-HOST-AQUI.turso.io?secure=true`
   - `TURSO_AUTH_TOKEN` = o token (mantém em segredo).
5. Faz **um** redeploy (ou “Redeploy” no último deployment) para instalar dependências e aplicar as variáveis.

A app já inclui o pacote `sqlalchemy-libsql` para esse URI. Em local continuas a usar `sqlite:///literature.db` por omissão. (Se instalares dependências em macOS, usa **Python 3.10 ou superior** para haver *wheel* do `libsql-experimental`; em 3.9 no Apple Silicon o `pip` pode tentar compilar a partir do código-fonte.)

### Projetos Vercel

Este repositório está pensado para **um único projeto** Vercel (um URL de produção). Cada `git push` ou `vercel deploy` cria **novas revisões** do *mesmo* projeto — não é necessário criar vários projetos para a mesma app.

---

## Formato do CSV

É necessário pelo menos **título** ou **resumo**. Nomes de coluna aceites (comparação flexível):

| Campo | Exemplos de nomes |
|--------|-------------------|
| Título | `title`, `paper title`, `document_title`, `article title`, `ti` |
| Ano | `year`, `publication year`, `pubyear`, `date`, `yr` |
| Resumo | `abstract`, `summary`, `ab`, `description` |

Outras colunas → metadados; na exportação aparecem como `extra::nome_da_coluna`. Colunas com `doi` no nome alimentam a busca por DOI; nomes como *journal*, *venue*, *cited by*, *citations*, etc., enriquecem a ficha na triagem.

---

## Destaques e padrões

- Vários termos: separar por vírgula, `;` ou nova linha.
- **Regex:** ativar a checkbox; expressões inválidas são ignoradas com aviso.
- Wildcards `*` e `?` no modo literal (com limites de palavra; ver implementação em `app.py` se precisares de detalhe fino).

---

## Rotas principais

| Rota | Método | Função |
|------|--------|--------|
| `/` | GET | Dataset ativo → triagem; senão → upload. |
| `/upload` | GET, POST | Lista datasets; POST importa CSV. |
| `/switch/<id>` | GET | Dataset ativo na sessão. |
| `/label/<id>` | GET | Próximo artigo sem rótulo. |
| `/label/submit/<article_id>` | POST | Sim / Não / Saltar + notas. |
| `/manage/<id>` | GET | Tabela, filtros, categorias. |
| `/highlights/<id>` | GET | Só resumos que casam com os destaques. |
| `/search/doi` | GET | `?doi=…` no dataset ativo. |
| `/export/<id>` | GET | CSV com decisões e extras. |
| `/works/<id>` | GET | Requer `templates/works.html` (pode estar em falta). |

---

## Base de dados legada

Se tiveres uma base SQLite **antiga** sem a coluna `datasets.search_query`, podes usar o script `test.py` como exemplo de `ALTER TABLE` (só nesse caso).

---

## Verificação rápida da instalação

```bash
python -c "from app import app, db; app.app_context().push(); print('OK', app.name)"
```

---

## Licença

MIT — ver [LICENSE](LICENSE).
