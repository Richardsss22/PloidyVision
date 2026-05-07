function cell_cycle_optimize_90()
    fprintf('\n=========================================================\n');
    fprintf(' OTIMIZADOR BRUTE-FORCE — Rumo aos 80%%+\n');
    fprintf('=========================================================\n');
    
    cfg = get_reference_config();
    feat = '1D Int. DAPI'; % O K-Means/ENS funciona melhor com esta feature base
    nc = cfg.nc;
    max_area = cfg.max_area;
    border_margin = cfg.border_margin;
    min_border_frac = cfg.min_border_frac;

    % 1. Carregamento dos caminhos
    basePath = '/Volumes/HDD 500GB/dados_sum_proj';
    datasets = cell(90, 3); 
    for i = 1:90
        datasets{i, 1} = sprintf('Tile %d', i);
        datasets{i, 2} = fullfile(basePath, sprintf('dapi_sum_tile_%d.tif', i));
        datasets{i, 3} = fullfile(basePath, sprintf('mask_max_tile_%d.tif', i));
    end

    % -------------------------------------------------------------
    % OTIMIZAÇÃO TOTAL: As 90 imagens!
    subset_tiles = 1:90; 
    % -------------------------------------------------------------

    % 2. Pré-calcular os Ground-Truths
    fprintf('A pré-calcular Ground-Truths das 90 imagens (Aguarde...)...\n');
    tsv_path = fullfile(basePath, 'GT_labels_40x_with_source_file.tsv');
    gt_table = readtable(tsv_path, 'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
    
    gt_masks = cell(90,1);
    todos_os_tiles_tsv = str2double(string(gt_table.tile_num));
    
    for d = subset_tiles
        instance_mask = imread(datasets{d,3});
        gt_img = zeros(size(instance_mask), 'double'); 
        linhas_da_tile = (todos_os_tiles_tsv == d);
        dados_tile = gt_table(linhas_da_tile, :);
        
        for idx = 1:height(dados_tile)
            cell_id = double(dados_tile.obj_num(idx)); 
            fase_texto = string(dados_tile.GT_label(idx)); 
            if contains(fase_texto, "G1", 'IgnoreCase', true), valor_fase = 1;
            elseif contains(fase_texto, "G2", 'IgnoreCase', true) || contains(fase_texto, "M", 'IgnoreCase', true), valor_fase = 2;
            else, valor_fase = 0; end
            gt_img(instance_mask == cell_id) = valor_fase;
        end
        gt_masks{d} = gt_img;
    end
    fprintf('GTs na memória! A iniciar Grid Search Massivo...\n\n');

    % 3. A Nova Grelha Expansiva (1620 combinações no total)
    min_area_vals = [300, 350, 400, 450, 500];  % 5 valores
    min_circ_vals = [0.55, 0.65, 0.75];         % 3 valores
    min_sol_vals  = [0.76, 0.80, 0.85];         % 3 valores
    min_ext_vals  = [0.25, 0.30, 0.35];         % 3 valores
    ws_foot_vals  = [1, 3, 5, 8];               % 4 valores
    min_int_vals  = [0.25, 0.35, 0.45];         % 3 valores
    
    best_score = -inf;   
    best_cfg = struct();
    
    total_combs = numel(min_area_vals) * numel(min_circ_vals) * numel(min_sol_vals) * ...
                  numel(min_ext_vals) * numel(ws_foot_vals) * numel(min_int_vals);
    fprintf('A testar %d combinações em %d imagens (Total: %d testes)...\n', ...
            total_combs, numel(subset_tiles), total_combs * numel(subset_tiles));

    comb_atual = 0;
    
    % Os 6 níveis do Inferno de Dante (Loops aninhados)
    for a = 1:numel(min_area_vals)
    for c = 1:numel(min_circ_vals)
    for s = 1:numel(min_sol_vals)
    for e = 1:numel(min_ext_vals)
    for w = 1:numel(ws_foot_vals)
    for ii = 1:numel(min_int_vals)
        comb_atual = comb_atual + 1;
        accs = nan(numel(subset_tiles),1);
        
        % Testar nas 90 lâminas
        for i_d = 1:numel(subset_tiles)
            d = subset_tiles(i_d);
            img = imread(datasets{d,2});
            gt = gt_masks{d};
            
            try
                r = analyse(img, nc, feat, ...
                    min_area_vals(a), max_area, ...
                    min_circ_vals(c), min_sol_vals(s), min_ext_vals(e), ...
                    min_int_vals(ii), ws_foot_vals(w), ...
                    border_margin, min_border_frac);
                
                % A nova avaliação rápida de GT
                ev = evaluate_mask_gt(r, r.ens, gt);
                accs(i_d) = 100 * ev.accuracy;
            catch
                accs(i_d) = NaN;
            end
        end
        
        avg_acc = mean(accs,'omitnan');
        min_acc = min(accs);
        
        % Imprime progresso a cada 20 combinações para não crachar o terminal
        if mod(comb_atual, 20) == 0
            fprintf('[%04d/%04d] Média Atual: %.2f%% | Recorde: %.2f%%\n', ...
                    comb_atual, total_combs, avg_acc, max(best_score, 0));
        end
        
        % Guardar a melhor configuração
        score = avg_acc; % Foco total em subir a média global
        
        if score > best_score
            best_score = score;
            best_cfg.min_area = min_area_vals(a);
            best_cfg.min_circ = min_circ_vals(c);
            best_cfg.min_sol  = min_sol_vals(s);
            best_cfg.min_ext  = min_ext_vals(e);
            best_cfg.ws_foot  = ws_foot_vals(w);
            best_cfg.min_int_frac = min_int_vals(ii);
            best_cfg.avg_acc = avg_acc;
            
            fprintf('=========================================================\n');
            fprintf('🏆 NOVO RECORDE: %.2f%%\n', avg_acc);
            fprintf('   Area: %d | Circ: %.2f | Sol: %.2f | Ext: %.2f | WS: %d | Int: %.2f\n', ...
                best_cfg.min_area, best_cfg.min_circ, best_cfg.min_sol, ...
                best_cfg.min_ext, best_cfg.ws_foot, best_cfg.min_int_frac);
            fprintf('=========================================================\n');
        end
    end
    end
    end
    end
    end
    end

    fprintf('\n=========================================================\n');
    fprintf(' OTIMIZAÇÃO CONCLUÍDA!\n');
    fprintf(' Melhor Média Global (Ensemble): %.2f%%\n', best_cfg.avg_acc);
    fprintf(' Substitui no get_reference_config():\n');
    fprintf(' cfg.min_area = %d;\n', best_cfg.min_area);
    fprintf(' cfg.min_circ = %.2f;\n', best_cfg.min_circ);
    fprintf(' cfg.min_sol  = %.2f;\n', best_cfg.min_sol);
    fprintf(' cfg.min_ext  = %.2f;\n', best_cfg.min_ext);
    fprintf(' cfg.ws_foot  = %d;\n', best_cfg.ws_foot);
    fprintf(' cfg.min_int_frac = %.2f;\n', best_cfg.min_int_frac);
    fprintf('=========================================================\n');
end