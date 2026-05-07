function run_benchmark()
cfg = get_reference_config();
datasets = {
    'Teste 1', 'dados-2/sub1/teste1.TIF', 'dados-2/sub1/ground-truth1.TIF';
    'Teste 2', 'dados-2/sub2/teste2.TIF', 'dados-2/sub2/ground-truth2.TIF';
    'Teste 3', 'dados-2/sub3/teste3.TIF', 'dados-2/sub3/ground-truth3.TIF';
    'Teste 4', 'dados-2/sub4/teste4.TIF', 'dados-2/sub4/ground-truths4.TIF';
};
feat_profiles = {
    '1D Int. DAPI', ...
    '1D Log(IntDAPI)', ...
    '2D Area + Mean', ...
    '2D Int + Mean', ...
    '2D LogInt + MeanNoBg', ...
    '2D Int + MeanNoBg', ...
    '2D Int + Area', ...
    '3D Int + Mean + Area', ...
    '2D IntDensity + Mean'
};
nc_vals = [2];
min_area        = cfg.min_area;
max_area        = cfg.max_area;
min_circ        = cfg.min_circ;
min_sol         = cfg.min_sol;
min_ext         = cfg.min_ext;
min_int         = cfg.min_int_frac;
ws_foot         = cfg.ws_foot;
border_margin   = cfg.border_margin;
min_border_frac = cfg.min_border_frac;
amb_thresh      = cfg.amb_thresh;
nbr_radius      = cfg.nbr_radius;
nF = numel(feat_profiles);
nD = size(datasets,1);

% Variáveis para médias globais
soma_km_acc  = zeros(nF, 1); soma_gm_acc  = zeros(nF, 1); soma_ens_acc = zeros(nF, 1);
soma_km_sil  = zeros(nF, 1); soma_gm_sil  = zeros(nF, 1); soma_ens_sil = zeros(nF, 1);
soma_km_g1   = zeros(nF, 1); soma_km_g2   = zeros(nF, 1); soma_amb     = zeros(nF, 1);
count_ok     = zeros(nF, 1);

% --- ACUMULADORES PARA A MATRIZ DE CONFUSÃO GLOBAL ---
global_gt_all = cell(nF, 1); % Guarda o Ground Truth de todas as 4 lâminas
global_pr_all = cell(nF, 1); % Guarda as Predições de todas as 4 lâminas
% -----------------------------------------------------

% Silencia os avisos laranjas do GMM 
warning('off', 'stats:gmdistribution:FailedToConvergeReps');
warning('off', 'stats:gmdistribution:FailedToConverge');
fprintf('\n========================================================================================================================\n');
fprintf(' BENCHMARK DE FEATURES — cell_cycle_v12 (4 Imagens Iniciais)\n');
fprintf('========================================================================================================================\n');

for d = 1:nD
    ds_name = datasets{d,1};
    img = imread(datasets{d,2});
    gt_img = imread(datasets{d,3});
    fprintf('\n>>>>> DATASET: %s\n', ds_name);
    for nc = nc_vals
        fprintf('\n--- %d FASES ---\n', nc);
        fprintf('%-25s | %8s %8s %8s | %8s %8s %8s | %5s %5s | %4s\n', ...
            'Feature', 'KM_Acc%', 'GM_Acc%', 'EN_Acc%', 'KM_Sil', 'GM_Sil', 'EN_Sil', 'KM_G1', 'KM_G2', 'Amb');
        fprintf('%s\n', repmat('-',1,115));
        for f = 1:nF
            feat = feat_profiles{f};
            try
                r = analyse(img, nc, feat, min_area, max_area, ...
                    min_circ, min_sol, min_ext, min_int, ...
                    ws_foot, border_margin, min_border_frac);
                
                r = add_silhouette_metrics(r);
                ev_km  = evaluate_against_gt(r, r.km,  gt_img, nc, amb_thresh, nbr_radius);
                ev_gm  = evaluate_against_gt(r, r.gm,  gt_img, nc, amb_thresh, nbr_radius);
                ev_ens = evaluate_against_gt(r, r.ens, gt_img, nc, amb_thresh, nbr_radius);
                
                km_sil_m = mean(r.sil_km,'omitnan');
                gm_sil_m = mean(r.sil_gm,'omitnan');
                ens_sil_m = mean(r.sil_ens,'omitnan');
                g1_count = sum(r.km==1);
                g2_count = sum(r.km==2);
                amb_count = ev_km.n_ambiguous;
                
                % Acumular totais para a tabela final de médias
                if ~isnan(ev_km.accuracy)
                    soma_km_acc(f) = soma_km_acc(f) + ev_km.accuracy;
                    soma_gm_acc(f) = soma_gm_acc(f) + ev_gm.accuracy;
                    soma_ens_acc(f) = soma_ens_acc(f) + ev_ens.accuracy;
                    
                    soma_km_sil(f) = soma_km_sil(f) + km_sil_m;
                    soma_gm_sil(f) = soma_gm_sil(f) + gm_sil_m;
                    soma_ens_sil(f) = soma_ens_sil(f) + ens_sil_m;
                    soma_km_g1(f) = soma_km_g1(f) + g1_count;
                    soma_km_g2(f) = soma_km_g2(f) + g2_count;
                    soma_amb(f)   = soma_amb(f) + amb_count;
                    count_ok(f) = count_ok(f) + 1;
                    
                    % --- EMPILHAR AS CÉLULAS PARA O GRÁFICO FINAL ---
                    global_gt_all{f} = [global_gt_all{f}; ev_ens.gt_labels(:)];
                    global_pr_all{f} = [global_pr_all{f}; ev_ens.pred_labels(:)];
                    % ------------------------------------------------
                end
                fprintf('%-25s | %7.2f%% %7.2f%% %7.2f%% | %8.4f %8.4f %8.4f | %5d %5d | %4d\n', ...
                    feat, ...
                    100*ev_km.accuracy, 100*ev_gm.accuracy, 100*ev_ens.accuracy, ...
                    km_sil_m, gm_sil_m, ens_sil_m, ...
                    g1_count, g2_count, amb_count);
            catch ME
                fprintf('%-25s | ERRO: %s\n', feat, ME.message);
            end
        end
    end
end
warning('on', 'stats:gmdistribution:FailedToConvergeReps');
warning('on', 'stats:gmdistribution:FailedToConverge');
fprintf('\n========================================================================================================================\n');
fprintf(' MÉDIAS GLOBAIS POR MÉTODO\n');
fprintf('========================================================================================================================\n');
fprintf('%-25s | %8s %8s %8s | %8s %8s %8s | %5s %5s | %4s\n', ...
    'Feature', 'KM_Acc%', 'GM_Acc%', 'EN_Acc%', 'KM_Sil', 'GM_Sil', 'EN_Sil', 'KM_G1', 'KM_G2', 'Amb');
fprintf('%s\n', repmat('-',1,115));
melhor_ens_acc = -1; melhor_ens_nome = '';
melhor_km_acc = -1;  melhor_km_nome = '';
melhor_gm_acc = -1;  melhor_gm_nome = '';
melhor_km_sil = -inf; melhor_kmsil_nome = '';
melhor_gm_sil = -inf; melhor_gmsil_nome = '';
melhor_ens_sil= -inf; melhor_enssil_nome= '';

for f = 1:nF
    if count_ok(f) > 0
        a_km  = 100 * soma_km_acc(f) / count_ok(f);
        a_gm  = 100 * soma_gm_acc(f) / count_ok(f);
        a_ens = 100 * soma_ens_acc(f) / count_ok(f);
        s_km  = soma_km_sil(f) / count_ok(f);
        s_gm  = soma_gm_sil(f) / count_ok(f);
        s_ens = soma_ens_sil(f) / count_ok(f);
        g1_m  = soma_km_g1(f) / count_ok(f);
        g2_m  = soma_km_g2(f) / count_ok(f);
        amb_m = soma_amb(f) / count_ok(f);
        fprintf('%-25s | %7.2f%% %7.2f%% %7.2f%% | %8.4f %8.4f %8.4f | %5.2f %5.2f | %4.2f\n', ...
            feat_profiles{f}, a_km, a_gm, a_ens, s_km, s_gm, s_ens, g1_m, g2_m, amb_m);
        
        if a_km > melhor_km_acc, melhor_km_acc = a_km; melhor_km_nome = feat_profiles{f}; end
        if a_gm > melhor_gm_acc, melhor_gm_acc = a_gm; melhor_gm_nome = feat_profiles{f}; end
        if a_ens > melhor_ens_acc, melhor_ens_acc = a_ens; melhor_ens_nome = feat_profiles{f}; end
        if s_km > melhor_km_sil, melhor_km_sil = s_km; melhor_kmsil_nome = feat_profiles{f}; end
        if s_gm > melhor_gm_sil, melhor_gm_sil = s_gm; melhor_gmsil_nome = feat_profiles{f}; end
        if s_ens > melhor_ens_sil, melhor_ens_sil = s_ens; melhor_enssil_nome = feat_profiles{f}; end
    else
        fprintf('%-25s | DADOS INSUFICIENTES\n', feat_profiles{f});
    end
end
fprintf('\n--- MELHORES PELAS MÉDIAS ---\n');
fprintf('Melhor KM accuracy:     %-25s (%.2f%%)\n', melhor_km_nome, melhor_km_acc);
fprintf('Melhor GM accuracy:     %-25s (%.2f%%)\n', melhor_gm_nome, melhor_gm_acc);
fprintf('Melhor ENS accuracy:    %-25s (%.2f%%)\n', melhor_ens_nome, melhor_ens_acc);
fprintf('Melhor KM silhouette:   %-25s (%.4f)\n', melhor_kmsil_nome, melhor_km_sil);
fprintf('Melhor GM silhouette:   %-25s (%.4f)\n', melhor_gmsil_nome, melhor_gm_sil);
fprintf('Melhor ENS silhouette:  %-25s (%.4f)\n', melhor_enssil_nome, melhor_ens_sil);

all_accs = [melhor_km_acc, melhor_gm_acc, melhor_ens_acc];
all_names = {melhor_km_nome, melhor_gm_nome, melhor_ens_nome};
[best_glob_acc, idx] = max(all_accs);
fprintf('Melhor feature global (melhor entre KM/GM/ENS): %s (%.2f%%)\n', all_names{idx}, best_glob_acc);
fprintf('========================================================================================================================\n\n');

% --- GERAR A MATRIZ DE CONFUSÃO GLOBAL NO FIM ---
idx_vencedor = find(strcmp(feat_profiles, melhor_ens_nome), 1);
if ~isempty(idx_vencedor)
    gt_final = global_gt_all{idx_vencedor};
    pr_final = global_pr_all{idx_vencedor};
    
    % Limpar os NaNs (Células ambíguas que o modelo descartou)
    validos = ~isnan(gt_final) & ~isnan(pr_final);
    gt_final = gt_final(validos);
    pr_final = pr_final(validos);
    
    % 1. Definir as classes consoante o número de fases (nc_vals)
    if nc_vals(1) == 3
        nomes_classes = {'G1', 'S', 'G2/M'};
        valores_classes = [1, 2, 3];
    else
        nomes_classes = {'G1', 'G2/M'};
        valores_classes = [1, 2];
    end
    
    % 2. Converter os números puros para categorias com texto
    gt_cat = categorical(gt_final, valores_classes, nomes_classes);
    pr_cat = categorical(pr_final, valores_classes, nomes_classes);
    
    % 3. Gerar o Gráfico já com as categorias certas
    figure('Name', 'Matriz de Confusão Global (Ensemble)', 'Color', 'w', 'Position', [400 300 550 450]);
    cm = confusionchart(gt_cat, pr_cat, ...
        'RowSummary','row-normalized', ...
        'ColumnSummary','column-normalized');
        
    % 4. Modificar a paleta de cores APENAS com HEX (Matlab Moderno)
    cm.DiagonalColor = '#7C7AAC';      % Azul escuro (Acertos)
    cm.OffDiagonalColor = '#d8d6ed';   % Azul muito claro/cinza (Erros)
    cm.FontColor = 'black';            % Força texto preto (opcional, para contraste)
    
    cm.Title = sprintf('Matriz de Confusão Global (4 Lâminas) - Ensemble\n%d Células Validadas (Feature: %s)', sum(validos), melhor_ens_nome);
end
% ------------------------------------------------
end