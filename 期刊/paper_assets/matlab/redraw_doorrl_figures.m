function redraw_doorrl_figures()
%REDRAW_DOORRL_FIGURES Generate paper-ready DOOR-RL figures with MATLAB.
%
% Run from any working directory:
%   redraw_doorrl_figures
%
% Outputs are written to:
%   <repo>/figures/paper/*.pdf
%   <repo>/figures/paper/*.png

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataDir = fullfile(rootDir, 'paper_assets', 'data');
outDir = fullfile(rootDir, 'figures', 'paper');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

setPaperStyle();

stage0 = readJSON(fullfile(dataDir, 'experiments__table3_fair_fix2_aggregate.json'));
nusc = readJSON(fullfile(dataDir, 'stage1_pilot_x__X_summary.json'));
nup20 = readJSON(fullfile(dataDir, 'nuplan_stage1_20k__summary.json'));
nup50 = readJSON(fullfile(dataDir, 'nuplan_stage1_50k__summary.json'));
planner = readJSON(fullfile(dataDir, 'nuplan_planner_sanity_50k__summary.json'));
subsets = readJSON(fullfile(dataDir, 'nuplan_interaction_subset_50k__summary.json'));
stats = readJSON(fullfile(dataDir, 'dataset_token_stats__summary.json'));

drawTeaser(outDir);
drawMethodOverview(outDir);
drawStage0SlotBudget(outDir);
if ~isOctave()
    drawStage0Metrics(stage0, outDir);
    drawStage1CrossDataset(nusc, nup20, nup50, outDir);
end
drawPlannerSubset(planner, subsets, outDir);
if ~isOctave()
    drawDatasetStats(stats, outDir);
end
drawSummaryComposite(stage0, nusc, nup20, nup50, subsets, stats, outDir);

fprintf('Wrote MATLAB-redrawn figures to %s\n', outDir);
end

function setPaperStyle()
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 9);
set(groot, 'defaultTextFontSize', 9);
set(groot, 'defaultAxesLineWidth', 0.8);
set(groot, 'defaultAxesBox', 'off');
set(groot, 'defaultAxesTickDir', 'out');
if ~isOctave()
    set(groot, 'defaultLegendBox', 'off');
end
end

function tf = isOctave()
tf = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end

function s = readJSON(path)
fid = fopen(path, 'r');
if fid < 0
    error('Cannot open JSON file: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
txt = fread(fid, '*char')';
s = jsondecode(txt);
end

function c = pal(key)
switch key
    case 'object'
        c = hex2rgb('#6B7280');
    case 'naive'
        c = hex2rgb('#D55E5E');
    case 'naiveVis'
        c = hex2rgb('#F2A65A');
    case 'dec'
        c = hex2rgb('#1F77B4');
    case 'decVis'
        c = hex2rgb('#2CA58D');
    case 'grey'
        c = hex2rgb('#9AA4B2');
    case 'dark'
        c = hex2rgb('#1F2937');
    case 'lightBlue'
        c = hex2rgb('#EFF6FF');
    case 'lightRed'
        c = hex2rgb('#FFF1F2');
    case 'lightGrey'
        c = hex2rgb('#F3F4F6');
    case 'purple'
        c = hex2rgb('#8B5CF6');
    case 'green'
        c = hex2rgb('#4CAF50');
    case 'orange'
        c = hex2rgb('#F39C34');
    case 'pink'
        c = hex2rgb('#D472B6');
    otherwise
        c = [0 0 0];
end
end

function rgb = hex2rgb(hex)
hex = erase(hex, '#');
rgb = [hex2dec(hex(1:2)), hex2dec(hex(3:4)), hex2dec(hex(5:6))] ./ 255;
end

function saveFig(fig, outDir, stem)
set(fig, 'Renderer', 'painters');
pdfPath = fullfile(outDir, [stem '.pdf']);
pngPath = fullfile(outDir, [stem '.png']);
if exist('exportgraphics', 'file') == 2
    exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(fig, pngPath, 'Resolution', 300, 'BackgroundColor', 'white');
else
    print(fig, pdfPath, '-dpdf', '-painters');
    print(fig, pngPath, '-dpng', '-r300');
end
close(fig);
end

function polishAxis(ax)
grid(ax, 'on');
set(ax, 'GridLineStyle', '--');
if ~isOctave()
    set(ax, 'GridAlpha', 0.25);
end
set(ax, 'XGrid', 'off');
set(ax, 'YGrid', 'on');
set(ax, 'Layer', 'top');
set(ax, 'LineWidth', 0.8);
end

function setBarColors(b, colors)
if isOctave()
    set(b, 'FaceColor', colors(1, :));
else
    b.CData = colors;
end
end

function drawTeaser(outDir)
fig = figure('Units', 'inches', 'Position', [1 1 12.8 4.8]);
ax = axes(fig, 'Position', [0 0 1 1]);
axis(ax, 'off');
hold(ax, 'on');

text(ax, 0.5, 0.94, 'Typed-budget relation-aware abstraction under a fixed latent budget', ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');

drawPanel(ax, [0.04 0.12 0.28 0.72], 'Failure: shared top-K', pal('lightRed'));
drawPanel(ax, [0.36 0.12 0.28 0.72], 'Fix: typed 12+4 budget', pal('lightBlue'));
drawPanel(ax, [0.68 0.12 0.28 0.72], 'Outcome: regime-dependent gain', [0.96 0.98 1.00]);

drawSceneGlyph(ax, 0.18, 0.48, true);
text(ax, 0.18, 0.23, 'relation tokens consume slots', 'HorizontalAlignment', 'center', 'Color', pal('naive'), 'FontWeight', 'bold');
text(ax, 0.18, 0.18, 'near-field dynamic agents are missed', 'HorizontalAlignment', 'center', 'Color', pal('naive'));

drawSceneGlyph(ax, 0.50, 0.48, false);
text(ax, 0.50, 0.23, 'K_{dyn}=12, K_{rel}=4', 'HorizontalAlignment', 'center', 'Color', pal('dec'), 'FontWeight', 'bold');
text(ax, 0.50, 0.18, 'same total 16-slot context', 'HorizontalAlignment', 'center', 'Color', pal('dec'));

annotation(fig, 'arrow', [0.325 0.36], [0.49 0.49], 'LineWidth', 1.4, 'Color', pal('dark'));
annotation(fig, 'arrow', [0.645 0.68], [0.49 0.49], 'LineWidth', 1.4, 'Color', pal('dark'));

barX = [0.74 0.84];
ret = [1.72 14.51];
coll = [0.610 0.259];
axes('Position', [0.72 0.45 0.20 0.22]);
b = bar(ret, 'FaceColor', 'flat');
setBarColors(b, [pal('object'); pal('dec')]);
set(gca, 'XTickLabel', {'Object', 'Dec-NoVis'});
ylabel('Return');
title('nuPlan 50k');
polishAxis(gca);

axes('Position', [0.72 0.18 0.20 0.20]);
b = bar(coll, 'FaceColor', 'flat');
setBarColors(b, [pal('object'); pal('dec')]);
set(gca, 'XTickLabel', {'Object', 'Dec-NoVis'});
ylabel('Collision');
polishAxis(gca);

text(ax, mean(barX), 0.79, 'nuScenes Stage 1: object-only stable', ...
    'HorizontalAlignment', 'center', 'Color', pal('object'), 'FontWeight', 'bold');
text(ax, mean(barX), 0.74, 'nuPlan 20k/50k: decoupled-no-vis wins', ...
    'HorizontalAlignment', 'center', 'Color', pal('dec'), 'FontWeight', 'bold');

saveFig(fig, outDir, 'fig1');
end

function drawMethodOverview(outDir)
fig = figure('Units', 'inches', 'Position', [1 1 13.2 5.0]);
ax = axes(fig, 'Position', [0 0 1 1]);
axis(ax, 'off');
hold(ax, 'on');

text(ax, 0.5, 0.94, 'DOOR-RL method overview', ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');

drawPanel(ax, [0.03 0.15 0.20 0.68], '1. Scene tokens', pal('lightGrey'));
drawPanel(ax, [0.27 0.15 0.20 0.68], '2. Shared top-K baseline', pal('lightRed'));
drawPanel(ax, [0.51 0.15 0.20 0.68], '3. Typed-budget abstraction', pal('lightBlue'));
drawPanel(ax, [0.75 0.15 0.22 0.68], '4. Latent imagination RL', [0.96 0.98 1.00]);

drawTokenStack(ax, 0.13, 0.57, {'EGO','DYN','DYN','REL','MAP','REL'}, ...
    [pal('naive'); pal('object'); pal('object'); pal('pink'); pal('purple'); pal('pink')]);
text(ax, 0.13, 0.28, 'oracle structured scene tokens', 'HorizontalAlignment', 'center');

drawSlotRow(ax, 0.37, 0.56, {'D','R','R','R','D','M','R','R'}, ...
    [pal('object'); pal('pink'); pal('pink'); pal('pink'); pal('object'); pal('purple'); pal('pink'); pal('pink')]);
text(ax, 0.37, 0.35, 'all token types compete for one K=16 bottleneck', ...
    'HorizontalAlignment', 'center', 'Color', pal('naive'));
text(ax, 0.37, 0.28, 'inter-type budget competition', ...
    'HorizontalAlignment', 'center', 'Color', pal('naive'), 'FontWeight', 'bold');

drawSlotRow(ax, 0.61, 0.62, {'D','D','D','D','D','D'}, repmat(pal('object'), 6, 1));
drawSlotRow(ax, 0.61, 0.49, {'R','R','R','R'}, repmat(pal('pink'), 4, 1));
text(ax, 0.61, 0.38, 'dynamic top-12 + relation top-4', ...
    'HorizontalAlignment', 'center', 'Color', pal('dec'), 'FontWeight', 'bold');
text(ax, 0.61, 0.30, 'concatenate to the same 16-slot context', ...
    'HorizontalAlignment', 'center', 'Color', pal('dec'));

drawBox(ax, [0.79 0.58 0.14 0.10], 'selected slots + action', [1 1 1], pal('dark'));
drawBox(ax, [0.79 0.43 0.14 0.10], 'world model', pal('lightBlue'), pal('dec'));
drawBox(ax, [0.79 0.28 0.14 0.10], 'actor / critic', [1 1 1], pal('dark'));
text(ax, 0.86, 0.21, 'next tokens, reward, collision, continue', ...
    'HorizontalAlignment', 'center');

annotation(fig, 'arrow', [0.235 0.27], [0.50 0.50], 'LineWidth', 1.2);
annotation(fig, 'arrow', [0.475 0.51], [0.50 0.50], 'LineWidth', 1.2);
annotation(fig, 'arrow', [0.715 0.75], [0.50 0.50], 'LineWidth', 1.2);
annotation(fig, 'arrow', [0.86 0.86], [0.58 0.53], 'LineWidth', 1.2);
annotation(fig, 'arrow', [0.86 0.86], [0.43 0.38], 'LineWidth', 1.2);

saveFig(fig, outDir, 'fig2');
end

function drawStage0SlotBudget(outDir)
% Slot composition is reconstructed from the documented aggregate notes.
% If a future slot-composition JSON is available, replace these values.
labels = {'Holistic-16','Object-Only','Naive Obj+Rel','Naive+Vis','Decoupled','Dec+Vis'};
ego =     [0 1.0 1.0 1.0 0.0 0.0];
dyn =     [0 15.0 3.7 5.5 12.0 12.0];
map =     [0 0.0 0.8 1.0 0.0 0.0];
rel =     [0 0.0 10.5 8.5 4.0 4.0];
learned = [16 0.0 0.0 0.0 0.0 0.0];
vals = [learned; ego; dyn; map; rel]';

fig = figure('Units', 'inches', 'Position', [1 1 12.5 4.8]);
ax = axes(fig);
b = bar(ax, vals, 'stacked', 'BarWidth', 0.72);
colors = [pal('grey'); pal('naive'); pal('object'); pal('purple'); pal('pink')];
for i = 1:numel(b)
    set(b(i), 'FaceColor', colors(i, :));
    set(b(i), 'EdgeColor', 'w');
    set(b(i), 'LineWidth', 0.7);
end
hold(ax, 'on');
if isOctave()
    plot(ax, [0.5, numel(labels) + 0.5], [16, 16], ':', 'Color', pal('dark'), 'LineWidth', 1.0);
    text(ax, numel(labels) + 0.35, 16.25, 'fair 16-slot budget', ...
        'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', pal('dark'));
else
    yline(ax, 16, ':', 'fair 16-slot budget', 'Color', pal('dark'), 'LineWidth', 1.0, ...
        'LabelHorizontalAlignment', 'right');
end
set(ax, 'XTick', 1:numel(labels), 'XTickLabel', labels);
xtickangle(ax, 12);
ylabel(ax, 'Average selected slots per sample');
ylim(ax, [0 18.5]);
title(ax, 'Stage-0 slot-type composition under the fair 16-slot budget');
legend(ax, {'learned/mixed','EGO','dynamic','map','relation'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');
polishAxis(ax);

text(ax, 3, 15.8, 'REL dominates shared top-K', ...
    'HorizontalAlignment', 'center', 'Color', pal('naive'), 'FontWeight', 'bold');
text(ax, 5.5, 15.8, 'typed 12+4 prevents slot starvation', ...
    'HorizontalAlignment', 'center', 'Color', pal('dec'), 'FontWeight', 'bold');

saveFig(fig, outDir, 'paper_stage0_slot_budget');
end

function drawStage0Metrics(stage0, outDir)
metrics = {'dyn_rollout_mse','rare_ade','interaction_recall_at_1m'};
titles = {'Dyn Rollout MSE','Rare ADE','Interaction Recall @ 1m'};
variants = {'object_only','object_relation','object_relation_decoupled','object_relation_decoupled_visibility'};
labels = {'Object','Naive','Dec','Dec+Vis'};
colors = [pal('object'); pal('naive'); pal('dec'); pal('decVis')];

fig = figure('Units', 'inches', 'Position', [1 1 12.6 4.0]);
t = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for m = 1:numel(metrics)
    ax = nexttile(t);
    means = zeros(1, numel(variants));
    errs = zeros(1, numel(variants));
    for i = 1:numel(variants)
        p = stage0.metrics.(variants{i}).(metrics{m});
        means(i) = p.mean;
        errs(i) = p.std;
    end
    b = bar(ax, means, 'FaceColor', 'flat', 'BarWidth', 0.72);
    setBarColors(b, colors);
    hold(ax, 'on');
    errorbar(ax, 1:numel(means), means, errs, 'k.', 'LineWidth', 1.0, 'CapSize', 6);
    set(ax, 'XTick', 1:numel(labels), 'XTickLabel', labels);
    xtickangle(ax, 15);
    title(ax, titles{m});
    if contains(metrics{m}, 'recall')
        ylabel(ax, 'higher is better');
        ylim(ax, [0.35 1.05]);
    else
        ylabel(ax, 'lower is better');
    end
    for i = 1:numel(means)
        text(ax, i, means(i) + max(errs) * 0.08 + max(means) * 0.015, sprintf('%.2f', means(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    panelTag(ax, char('a' + m - 1));
    polishAxis(ax);
end
title(t, 'Stage-0 fair-budget representation metrics');
saveFig(fig, outDir, 'paper_stage0_metrics');
end

function drawStage1CrossDataset(nusc, nup20, nup50, outDir)
datasets = {'nuScenes','nuPlan 20k','nuPlan 50k'};
sources = {nusc, nup20, nup50};
methods = {'wm_object','wm_decoupled','wm_decoupled_no_vis'};
methodLabels = {'Object','Dec','Dec-NoVis'};
colors = [pal('object'); pal('decVis'); pal('dec')];

fig = figure('Units', 'inches', 'Position', [1 1 11.8 4.3]);
t = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
metricNames = {'latent_return_mean','imagined_collision_rate'};
ylabels = {'Imagined return','Imagined collision rate'};

for mm = 1:2
    ax = nexttile(t);
    hold(ax, 'on');
    for k = 1:numel(methods)
        xs = [];
        means = [];
        errs = [];
        for d = 1:numel(sources)
            if hasCondition(sources{d}, methods{k})
                [mu, sd] = metricPoint(sources{d}, methods{k}, metricNames{mm});
                xs(end+1) = d; %#ok<AGROW>
                means(end+1) = mu; %#ok<AGROW>
                errs(end+1) = sd; %#ok<AGROW>
            end
        end
        errorbar(ax, xs, means, errs, '-o', 'Color', colors(k, :), ...
            'MarkerFaceColor', colors(k, :), 'LineWidth', 1.8, 'CapSize', 5);
    end
    set(ax, 'XTick', 1:numel(datasets), 'XTickLabel', datasets);
    ylabel(ax, ylabels{mm});
    title(ax, ylabels{mm});
    xlim(ax, [0.7 3.3]);
    panelTag(ax, char('a' + mm - 1));
    polishAxis(ax);
end
legend(nexttile(t, 2), methodLabels, 'Location', 'northoutside', 'Orientation', 'horizontal');
title(t, 'Stage-1 cross-dataset ranking reversal');
saveFig(fig, outDir, 'paper_stage1_cross_dataset');
end

function drawPlannerSubset(planner, subsets, outDir)
fig = figure('Units', 'inches', 'Position', [1 1 12.2 4.7]);
if isOctave()
    ax = subplot(1, 2, 1);
else
    t = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    ax = nexttile(t);
end
metrics = {'teacher_action_mse','latent_return_mean','imagined_collision_rate'};
metricLabels = {'Teacher action MSE','Return','Collision rate'};
y = numel(metrics):-1:1;
hold(ax, 'on');
for i = 1:numel(metrics)
    obj = planner.summary.wm_object.mean.(metrics{i});
    dec = planner.summary.wm_decoupled_no_vis.mean.(metrics{i});
    plot(ax, [obj dec], [y(i) y(i)], '-', 'Color', [0.78 0.80 0.84], 'LineWidth', 2);
    scatter(ax, obj, y(i), 65, pal('object'), 'filled', 'MarkerEdgeColor', 'w');
    scatter(ax, dec, y(i), 65, pal('dec'), 'filled', 'MarkerEdgeColor', 'w');
end
set(ax, 'YTick', y, 'YTickLabel', metricLabels);
xlabel(ax, 'Metric value');
title(ax, 'nuPlan 50k planner-like sanity');
legend(ax, {'','Object','Dec-NoVis'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
panelTag(ax, 'a');
polishAxis(ax);

if isOctave()
    ax = subplot(1, 2, 2);
else
    ax = nexttile(t);
end
subsetKeys = {'lane_conflict','low_ttc_proxy','rare_agent_dense','dense_agents','high_interaction_union'};
subsetLabels = {'Lane conflict','Low TTC','Rare+dense','Dense','Interaction union'};
obj = zeros(1, numel(subsetKeys));
dec = zeros(1, numel(subsetKeys));
for i = 1:numel(subsetKeys)
    obj(i) = subsets.summary.wm_object.(subsetKeys{i}).imagined_collision_rate_mean;
    dec(i) = subsets.summary.wm_decoupled_no_vis.(subsetKeys{i}).imagined_collision_rate_mean;
end
yy = 1:numel(subsetKeys);
barh(ax, yy + 0.18, obj, 0.34, 'FaceColor', pal('object'), 'EdgeColor', 'none');
hold(ax, 'on');
barh(ax, yy - 0.18, dec, 0.34, 'FaceColor', pal('dec'), 'EdgeColor', 'none');
set(ax, 'YTick', yy, 'YTickLabel', subsetLabels);
set(ax, 'YDir', 'reverse');
xlabel(ax, 'Imagined collision rate');
title(ax, 'Interaction-conditioned subsets');
legend(ax, {'Object','Dec-NoVis'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
for i = 1:numel(subsetKeys)
    text(ax, max(obj(i), dec(i)) + 0.015, yy(i), sprintf('\\Delta %.2f', obj(i) - dec(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 8);
end
panelTag(ax, 'b');
polishAxis(ax);

if ~isOctave()
    title(t, 'Downstream offline evidence on nuPlan 50k');
end
saveFig(fig, outDir, 'paper_planner_subset_summary');
end

function drawDatasetStats(stats, outDir)
metrics = {'dynamic_tokens_per_sample','rare_tokens_per_sample','visibility_dynamic','teacher_action_l2'};
labels = {'Dynamic tokens','Rare tokens','Visibility','Teacher action L2'};
nusc = stats.nuscenes.metrics;
nup = stats.nuplan_50k.metrics;

fig = figure('Units', 'inches', 'Position', [1 1 12.0 3.7]);
t = tiledlayout(fig, 1, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:numel(metrics)
    ax = nexttile(t);
    vals = [nusc.(metrics{i}).mean, nup.(metrics{i}).mean];
    errs = [nusc.(metrics{i}).std, nup.(metrics{i}).std];
    b = bar(ax, vals, 'FaceColor', 'flat', 'BarWidth', 0.65);
    setBarColors(b, [pal('grey'); pal('dec')]);
    hold(ax, 'on');
    errorbar(ax, 1:2, vals, errs, 'k.', 'LineWidth', 0.9, 'CapSize', 5);
    set(ax, 'XTick', 1:2, 'XTickLabel', {'nuScenes','nuPlan'});
    title(ax, labels{i});
    for j = 1:2
        text(ax, j, vals(j) + max(errs) * 0.05 + max(vals) * 0.03, sprintf('%.2f', vals(j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    panelTag(ax, char('a' + i - 1));
    polishAxis(ax);
end
title(t, 'Dataset-statistics context for the ranking reversal');
saveFig(fig, outDir, 'paper_dataset_stats');
end

function drawSummaryComposite(stage0, nusc, nup20, nup50, subsets, stats, outDir)
fig = figure('Units', 'inches', 'Position', [1 1 12.8 8.0]);
if isOctave()
    ax = subplot(2, 2, 1);
else
    t = tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    ax = nexttile(t);
end
variants = {'object_only','object_relation','object_relation_decoupled','object_relation_decoupled_visibility'};
labels = {'Object','Naive','Dec','Dec+Vis'};
colors = [pal('object'); pal('naive'); pal('dec'); pal('decVis')];
means = zeros(1, 4);
errs = zeros(1, 4);
for i = 1:4
    p = stage0.metrics.(variants{i}).interaction_recall_at_1m;
    means(i) = p.mean;
    errs(i) = p.std;
end
b = bar(ax, means, 'FaceColor', 'flat', 'BarWidth', 0.7);
setBarColors(b, colors);
hold(ax, 'on');
if isOctave()
    plot(ax, 1:4, means, 'k.', 'MarkerSize', 10);
else
    errorbar(ax, 1:4, means, errs, 'k.', 'LineWidth', 1, 'CapSize', 5);
end
set(ax, 'XTick', 1:4, 'XTickLabel', labels);
ylabel(ax, 'IntRec@1m');
ylim(ax, [0.35 1.05]);
title(ax, 'Stage 0: shared mixing collapses');
panelTag(ax, 'a');
polishAxis(ax);

if isOctave()
    ax = subplot(2, 2, 2);
else
    ax = nexttile(t);
end
datasets = {'nuScenes','20k','50k'};
src = {nusc, nup20, nup50};
for method = {'wm_object','wm_decoupled_no_vis'}
    key = method{1};
    yv = zeros(1, 3);
    ev = zeros(1, 3);
    for d = 1:3
        [yv(d), ev(d)] = metricPoint(src{d}, key, 'latent_return_mean');
    end
    color = pal('object');
    if strcmp(key, 'wm_decoupled_no_vis')
        color = pal('dec');
    end
    hold(ax, 'on');
    if isOctave()
        plot(ax, 1:3, yv, '-o', 'Color', color, ...
            'MarkerFaceColor', color, 'LineWidth', 1.8);
    else
        errorbar(ax, 1:3, yv, ev, '-o', 'Color', color, ...
            'MarkerFaceColor', color, 'LineWidth', 1.8, 'CapSize', 5);
    end
end
set(ax, 'XTick', 1:3, 'XTickLabel', datasets);
ylabel(ax, 'Imagined return');
title(ax, 'Stage 1: ranking reversal');
legend(ax, {'Object','Dec-NoVis'}, 'Location', 'northwest');
panelTag(ax, 'b');
polishAxis(ax);

if isOctave()
    ax = subplot(2, 2, 3);
else
    ax = nexttile(t);
end
subsetKeys = {'lane_conflict','low_ttc_proxy','high_interaction_union'};
subsetLabels = {'Lane conflict','Low TTC','Interaction union'};
obj = zeros(1, 3);
dec = zeros(1, 3);
for i = 1:3
    obj(i) = subsets.summary.wm_object.(subsetKeys{i}).imagined_collision_rate_mean;
    dec(i) = subsets.summary.wm_decoupled_no_vis.(subsetKeys{i}).imagined_collision_rate_mean;
end
x = 1:3;
w = 0.34;
bar(ax, x - w/2, obj, w, 'FaceColor', pal('object'), 'EdgeColor', 'none');
hold(ax, 'on');
bar(ax, x + w/2, dec, w, 'FaceColor', pal('dec'), 'EdgeColor', 'none');
set(ax, 'XTick', x, 'XTickLabel', subsetLabels);
ylabel(ax, 'Collision rate');
title(ax, 'nuPlan interaction-heavy subsets');
legend(ax, {'Object','Dec-NoVis'}, 'Location', 'northeast');
panelTag(ax, 'c');
polishAxis(ax);

if isOctave()
    ax = subplot(2, 2, 4);
else
    ax = nexttile(t);
end
metricKeys = {'dynamic_tokens_per_sample','rare_tokens_per_sample','teacher_action_l2'};
metricLabels = {'Dyn','Rare','Action L2'};
nuscStats = stats.nuscenes.metrics;
nupStats = stats.nuplan_50k.metrics;
nuscVals = zeros(1, 3);
nupVals = zeros(1, 3);
for i = 1:3
    nuscVals(i) = nuscStats.(metricKeys{i}).mean;
    nupVals(i) = nupStats.(metricKeys{i}).mean;
end
x = 1:3;
bar(ax, x - w/2, nuscVals, w, 'FaceColor', pal('grey'), 'EdgeColor', 'none');
hold(ax, 'on');
bar(ax, x + w/2, nupVals, w, 'FaceColor', pal('dec'), 'EdgeColor', 'none');
set(ax, 'XTick', x, 'XTickLabel', metricLabels);
title(ax, 'Dataset context');
legend(ax, {'nuScenes','nuPlan'}, 'Location', 'northwest');
panelTag(ax, 'd');
polishAxis(ax);

if ~isOctave()
    title(t, 'Compact summary of DOOR-RL evidence');
end
saveFig(fig, outDir, 'paper_summary_charts');
end

function tf = hasCondition(s, condition)
if isfield(s, 'conditions')
    tf = isfield(s.conditions, condition);
else
    tf = isfield(s, condition);
end
end

function [mu, sd] = metricPoint(s, condition, metric)
if isfield(s, 'conditions')
    mu = s.conditions.(condition).mean.(metric);
    sd = s.conditions.(condition).std_across_seeds.(metric);
else
    mu = s.(condition).(metric).mean;
    sd = s.(condition).(metric).std;
end
end

function panelTag(ax, tag)
text(ax, -0.12, 1.08, ['(' tag ')'], 'Units', 'normalized', ...
    'FontWeight', 'bold', 'FontSize', 11, 'VerticalAlignment', 'bottom');
end

function drawPanel(ax, pos, titleText, faceColor)
rectangle(ax, 'Position', pos, 'Curvature', 0.04, ...
    'FaceColor', faceColor, 'EdgeColor', [0.78 0.82 0.88], 'LineWidth', 1.0);
text(ax, pos(1) + pos(3)/2, pos(2) + pos(4) - 0.06, titleText, ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
end

function drawBox(ax, pos, txt, faceColor, edgeColor)
rectangle(ax, 'Position', pos, 'Curvature', 0.06, ...
    'FaceColor', faceColor, 'EdgeColor', edgeColor, 'LineWidth', 1.0);
text(ax, pos(1) + pos(3)/2, pos(2) + pos(4)/2, txt, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
end

function drawSceneGlyph(ax, cx, cy, failureMode)
theta = linspace(0, 2*pi, 100);
plot(ax, cx + 0.115*cos(theta), cy + 0.115*sin(theta), ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.1);
rectangle(ax, 'Position', [cx-0.018 cy-0.025 0.036 0.05], 'Curvature', 0.2, ...
    'FaceColor', pal('dark'), 'EdgeColor', 'w');
text(ax, cx, cy-0.055, 'ego', 'HorizontalAlignment', 'center', 'FontSize', 8);
pts = [-0.075 0.060; 0.075 0.050; -0.055 -0.070; 0.090 -0.050; 0.010 0.095];
for i = 1:size(pts, 1)
    isMissed = failureMode && any(i == [1 3 5]);
    c = pal('object');
    edge = 'w';
    if isMissed
        c = [1 1 1];
        edge = pal('naive');
    end
    rectangle(ax, 'Position', [cx+pts(i,1)-0.014 cy+pts(i,2)-0.014 0.028 0.028], ...
        'Curvature', 0.5, 'FaceColor', c, 'EdgeColor', edge, 'LineWidth', 1.2);
end
for i = 1:5
    ang = i * 2*pi/5 + 0.2;
    rectangle(ax, 'Position', [cx+0.145*cos(ang)-0.012 cy+0.145*sin(ang)-0.012 0.024 0.024], ...
        'Curvature', 0.5, 'FaceColor', pal('pink'), 'EdgeColor', 'w');
end
end

function drawTokenStack(ax, cx, cy, txts, colors)
n = numel(txts);
h = 0.045;
w = 0.10;
for i = 1:n
    y = cy + (n/2 - i) * (h + 0.008);
    rectangle(ax, 'Position', [cx-w/2 y w h], 'Curvature', 0.08, ...
        'FaceColor', colors(i, :), 'EdgeColor', 'w');
    text(ax, cx, y + h/2, txts{i}, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 8);
end
end

function drawSlotRow(ax, cx, cy, txts, colors)
n = numel(txts);
w = 0.034;
h = 0.050;
gap = 0.006;
total = n*w + (n-1)*gap;
x0 = cx - total/2;
for i = 1:n
    x = x0 + (i-1)*(w+gap);
    rectangle(ax, 'Position', [x cy w h], 'Curvature', 0.08, ...
        'FaceColor', colors(i, :), 'EdgeColor', 'w');
    text(ax, x + w/2, cy + h/2, txts{i}, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 8);
end
end
