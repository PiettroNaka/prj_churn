# windows/

Para quem está no Windows e não quer abrir terminal. **Dê duplo clique** no
arquivo — a janela preta abre, faz o serviço e fica aberta com a mensagem.

| arquivo | o que faz |
|---|---|
| `ver-runs.bat` | gera o relatório dos experimentos e abre no navegador (HTML estático, sem servidor) |
| `abrir-mlflow.bat` | prepara o que faltar e sobe a UI do MLflow em `localhost:5000` — deixe a janela aberta enquanto usa |
| `atualizar.bat` | traz a versão mais nova do GitHub e desfaz qualquer enrosco do Git |

Os três sobem para a raiz do projeto antes de rodar (`cd /d "%~dp0.."`), então
não importa de onde foram chamados — o erro clássico de "abri na pasta errada"
não acontece.

> **`atualizar.bat` descarta o que estiver modificado na pasta.** É de propósito:
> aqui o projeto é lido, não escrito. Se você mexeu em algum arquivo e quer
> guardar, copie para fora antes de rodar.

Se algo falhar, a mensagem fica na tela — copie e peça ajuda. O passo a passo
completo, com o caminho pelo terminal, está em [`docs/rodar-local.md`](../docs/rodar-local.md).
