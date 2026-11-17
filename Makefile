# Atalhos para instalar, baixar dados, reproduzir experimentos e compilar o artigo.
# Novas execuções gravam em saidas/ para não sobrescrever os resultados oficiais.

PY = poetry run python
EXP = experimentos
OUT = saidas

.PHONY: verificar instalar dados e01 e02 e03 e04 e05 e06 e07 e08 e09 e10 e11 e12 e13 e14 e15 e16 e17 artigo lint limpar

instalar:
	poetry install --all-extras

dados:
	$(PY) scripts/baixar_dados.py

$(OUT):
	mkdir -p $(OUT)

e01: ; $(PY) $(EXP)/E01-criterio-espectral/criterio_espectral.py
e02: ; $(PY) $(EXP)/E02-fourier-grupo-errado/fourier_grupo_errado.py
e03: ; $(PY) $(EXP)/E03-memoria-superposicao/memoria_superposicao.py
e04: | $(OUT)
	$(PY) $(EXP)/E04-microteste-energia/microteste_energia.py dados/glove-100-angular.hdf5 $(OUT)/microteste-resultados.json
	$(PY) $(EXP)/E04-microteste-energia/microteste_confirmacao.py dados/glove-100-angular.hdf5 $(OUT)/microteste-confirmacao.json
e05: | $(OUT) ; $(PY) $(EXP)/E05-lei-de-escala-memoria/lei_de_escala_memoria.py $(OUT)/lei-de-escala.json
e06: | $(OUT) ; $(PY) $(EXP)/E06-comunicacao-gpt2/comunicacao_gpt2.py dados/moby.txt $(OUT)/comunicacao-gpt2.json
e07: | $(OUT) ; $(PY) $(EXP)/E07-atencao-esparsa-gpt2/atencao_esparsa_gpt2.py dados/moby.txt $(OUT)/atencao-esparsa-gpt2.json
e08: | $(OUT) ; $(PY) $(EXP)/E08-atencao-escala-e-proxy/atencao_escala_e_proxy.py dados/moby.txt $(OUT)/atencao-escala-e-proxy.json
e09: | $(OUT) ; $(PY) $(EXP)/E09-expoente-texto-embaralhado/expoente_texto_embaralhado.py dados/moby.txt $(OUT)/expoente-texto-embaralhado.json
e10: | $(OUT) ; $(PY) $(EXP)/E10-treino-com-orcamento/treino_com_orcamento.py dados/moby.txt $(OUT)/treino-com-orcamento.json 2000 2
e11: | $(OUT) ; $(PY) $(EXP)/E11-canais-de-gradiente/canais_de_gradiente.py dados/fashion $(OUT)/canais-de-gradiente.json
e12: | $(OUT) ; cd $(EXP)/E12-canal-quantizado-e-ruidoso && $(PY) canal_quantizado.py rodar ../../dados/fashion ../../$(OUT)/e12-s0.json 0 && $(PY) canal_quantizado.py rodar ../../dados/fashion ../../$(OUT)/e12-s1.json 1 && $(PY) canal_quantizado.py analisar ../../$(OUT)/e12.json ../../$(OUT)/e12-s0.json ../../$(OUT)/e12-s1.json --e11 ../E11-canais-de-gradiente/resultados/canais-de-gradiente.json

e13: | $(OUT)
	$(PY) $(EXP)/E13-fatorial-de-I/fatorial_de_I.py referencia dados/fashion $(OUT)/e13-ref.json
	for s in 0 1 2; do $(PY) $(EXP)/E13-fatorial-de-I/fatorial_de_I.py rodar dados/fashion $(OUT)/e13-ref.json $(OUT)/e13-s$$s.json $$s; done
	$(PY) $(EXP)/E13-fatorial-de-I/fatorial_de_I.py analisar $(OUT)/e13.json $(OUT)/e13-ref.json $(OUT)/e13-s0.json $(OUT)/e13-s1.json $(OUT)/e13-s2.json
e14: | $(OUT)
	for s in 0 1; do $(PY) $(EXP)/E14-perturbacao-real/perturbacao_real.py rodar dados/fashion $(OUT)/e14-s$$s.json $$s; done
	$(PY) $(EXP)/E14-perturbacao-real/perturbacao_real.py analisar $(OUT)/e14.json $(OUT)/e14-s0.json $(OUT)/e14-s1.json --e11 $(EXP)/E11-canais-de-gradiente/resultados/canais-de-gradiente.json
e15: | $(OUT)
	for s in 0 1; do $(PY) $(EXP)/E15-ruido-do-minilote/ruido_do_minilote.py rodar dados/fashion $(OUT)/e15-s$$s.json $$s; done
	$(PY) $(EXP)/E15-ruido-do-minilote/ruido_do_minilote.py analisar $(OUT)/e15.json $(OUT)/e15-s0.json $(OUT)/e15-s1.json

e16: | $(OUT)
	for s in 0 1; do $(PY) $(EXP)/E16-lote-grande/lote_grande.py rodar dados/fashion $(OUT)/e16-s$$s.json $$s; done
	$(PY) $(EXP)/E16-lote-grande/lote_grande.py analisar $(OUT)/e16.json $(EXP)/E15-ruido-do-minilote/resultados/ruido-do-minilote.json $(OUT)/e16-s0.json $(OUT)/e16-s1.json

e17: | $(OUT)
	for s in 0 1 2; do $(PY) $(EXP)/E17-cache-sinaptico/cache_sinaptico.py rodar dados/fashion $(OUT)/e17-s$$s.json $$s; done
	$(PY) $(EXP)/E17-cache-sinaptico/cache_sinaptico.py analisar $(OUT)/e17.json $(OUT)/e17-s0.json $(OUT)/e17-s1.json $(OUT)/e17-s2.json

verificar:
	python3 scripts/verificar_ensaio.py

artigo:
	python3 scripts/gerar_artigo.py
	cd artigo && tectonic main.tex

lint:
	poetry run ruff check $(EXP) scripts

limpar:
	rm -rf $(OUT) artigo/*.aux artigo/*.log artigo/*.out
