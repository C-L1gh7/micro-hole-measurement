function generate_figures()
    % 生成图片：拟合曲线、函数对比、3D形貌复原（2张）、景深合成
    
    % 配色方案
    colors = struct();
    colors.primary = [0.2, 0.4, 0.8];      % 深蓝色
    colors.secondary = [0.9, 0.3, 0.2];    % 红色
    colors.accent1 = [0.2, 0.7, 0.5];      % 青绿色
    colors.accent2 = [0.95, 0.6, 0.1];     % 橙色
    colors.gray = [0.5, 0.5, 0.5];         % 灰色
    colors.light_blue = [0.4, 0.7, 0.95];  % 浅蓝色
    
    % 选择数据文件夹
    base_path = uigetdir('', '请选择包含processed和original文件夹的目录');
    if base_path == 0
        disp('用户取消操作');
        return;
    end
    
    % 创建输出文件夹
    output_dir = fullfile(base_path, 'ppt_figures');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % 分析数据
    processed_path = fullfile(base_path, 'processed');
    original_path = fullfile(base_path, 'original');
    top_folder = fullfile(processed_path, 'top');
    bottom_folder = fullfile(processed_path, 'bottom');
    
    fprintf('正在分析数据...\n');
    top_result = analyze_folder_focus(top_folder);
    bottom_result = analyze_folder_focus(bottom_folder);
    
    if isempty(top_result) || isempty(bottom_result)
        error('数据分析失败，请检查文件夹结构');
    end
    
    % 生成图表
    fprintf('正在生成图表...\n\n');
    
    % 图1: 拟合曲线对比图
    fprintf('1. 生成拟合曲线对比图...\n');
    fig1 = figure('Position', [100, 100, 1200, 800], 'Color', 'w');
    generate_fitting_curves(top_result, bottom_result, colors);
    saveas(fig1, fullfile(output_dir, 'Fig1_Fitting_Curves.png'));
    saveas(fig1, fullfile(output_dir, 'Fig1_Fitting_Curves.svg'));
    
    % 图2: Lorentzian vs Gaussian 对比
    fprintf('2. 生成拟合函数对比图...\n');
    fig2 = figure('Position', [100, 100, 1000, 600], 'Color', 'w');
    generate_function_comparison(top_result, colors);
    saveas(fig2, fullfile(output_dir, 'Fig2_Function_Comparison.png'));
    saveas(fig2, fullfile(output_dir, 'Fig2_Function_Comparison.svg'));
    
    % 图3: 3D形貌复原图
    fprintf('3. 生成3D微孔形貌复原图（这可能需要几分钟）...\n');
    
    % 先计算一次深度图（供三个函数共用）
    [image_files, indices] = get_sorted_images(original_path);
    first_img = imread(fullfile(original_path, image_files(1).name));
    if size(first_img, 3) == 3
        gray_img = rgb2gray(first_img);
    else
        gray_img = first_img;
    end
    [height, width] = size(gray_img);
    [depth_map, focus_map] = calculate_depth_map(original_path, image_files, indices, height, width);
    
    % 图3A: 景深合成图
    generate_focus_stacked_image(original_path, depth_map, focus_map, colors, output_dir, image_files, indices);
    
    % 图3B: Jet色图3D模型
    generate_3d_jet_model(depth_map, top_result, bottom_result, colors, output_dir, height, width);
    
    % 图3C: RGB纹理3D模型
    generate_3d_rgb_textured_model(original_path, depth_map, top_result, bottom_result, colors, output_dir, image_files, indices, height, width);
    
    fprintf('\n所有图表已保存到: %s\n', output_dir);
    fprintf('共生成5组图表（PNG和SVG格式）\n');
end

%% 图1: 拟合曲线对比图
function generate_fitting_curves(top_result, bottom_result, colors)
    % TOP曲线
    subplot(1, 2, 1);
    hold on; box on; grid on;
    
    % 实测数据点
    scatter(top_result.indices, top_result.focus_measures, 80, ...
        'MarkerFaceColor', colors.primary, 'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Measured data');
    
    % 拟合曲线
    plot(top_result.x_smooth, top_result.y_smooth, '-', ...
        'Color', colors.secondary, 'LineWidth', 3, 'DisplayName', 'Lorentzian fit');
    
    % 峰值标记
    plot(top_result.peak_index, max(top_result.y_smooth), 'p', ...
        'MarkerSize', 18, 'MarkerFaceColor', colors.accent2, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Peak');
    
    % 半高线
    half_height = max(top_result.y_smooth) / 2;
    plot([min(top_result.x_smooth), max(top_result.x_smooth)], ...
        [half_height, half_height], '--', 'Color', colors.accent1, ...
        'LineWidth', 2, 'DisplayName', 'FWHM');
    
    xlabel('Image Index', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Focus Measure', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Top Surface\nPeak = %.3f, R² = %.3f', ...
        top_result.peak_index, top_result.r_squared), ...
        'FontSize', 16, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 11);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    
    % BOTTOM曲线
    subplot(1, 2, 2);
    hold on; box on; grid on;
    
    scatter(bottom_result.indices, bottom_result.focus_measures, 80, ...
        'MarkerFaceColor', colors.primary, 'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Measured data');
    
    plot(bottom_result.x_smooth, bottom_result.y_smooth, '-', ...
        'Color', colors.secondary, 'LineWidth', 3, 'DisplayName', 'Lorentzian fit');
    
    plot(bottom_result.peak_index, max(bottom_result.y_smooth), 'p', ...
        'MarkerSize', 18, 'MarkerFaceColor', colors.accent2, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Peak');
    
    half_height = max(bottom_result.y_smooth) / 2;
    plot([min(bottom_result.x_smooth), max(bottom_result.x_smooth)], ...
        [half_height, half_height], '--', 'Color', colors.accent1, ...
        'LineWidth', 2, 'DisplayName', 'FWHM');
    
    xlabel('Image Index', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Focus Measure', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Bottom Surface\nPeak = %.3f, R² = %.3f', ...
        bottom_result.peak_index, bottom_result.r_squared), ...
        'FontSize', 16, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 11);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
end

%% 图2: Lorentzian vs Gaussian对比（精简版）
function generate_function_comparison(result, colors)
    % 使用TOP数据同时拟合Lorentzian和Gaussian
    
    % Lorentzian拟合（已有）
    lorentz_fit = result.y_smooth;
    
    % Gaussian拟合
    gaussian_params = fit_gaussian(result.indices, result.focus_measures);
    x_smooth = result.x_smooth;
    gaussian_fit = gaussian_params(1) * exp(-((x_smooth - gaussian_params(2)).^2) / (2 * gaussian_params(3)^2));
    
    % 计算Gaussian的R²
    y_pred_gaussian = gaussian_params(1) * exp(-((result.indices - gaussian_params(2)).^2) / (2 * gaussian_params(3)^2));
    r2_gaussian = calculate_r_squared(result.focus_measures, y_pred_gaussian);
    
    % 绘图
    hold on; box on; grid on;
    
    % 实测数据
    scatter(result.indices, result.focus_measures, 100, ...
        'MarkerFaceColor', colors.gray, 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, 'DisplayName', 'Data');
    
    % Lorentzian拟合
    plot(x_smooth, lorentz_fit, '-', 'Color', colors.secondary, ...
        'LineWidth', 3, 'DisplayName', sprintf('Lorentzian (R²=%.3f)', result.r_squared));
    
    % Gaussian拟合
    plot(x_smooth, gaussian_fit, '--', 'Color', colors.primary, ...
        'LineWidth', 3, 'DisplayName', sprintf('Gaussian (R²=%.3f)', r2_gaussian));
    
    xlabel('Image Index', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Focus Measure', 'FontSize', 14, 'FontWeight', 'bold');
    title('Fitting Comparison', 'FontSize', 16, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 12);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
end

%% 图3A: 景深合成的全聚焦平面图
function generate_focus_stacked_image(original_path, depth_map, focus_map, colors, output_dir, image_files, indices)
    fprintf('   3A. 生成景深合成的全聚焦图像...\n');
    
    % 读取第一张图像确定尺寸
    first_img = imread(fullfile(original_path, image_files(1).name));
    [height, width, channels] = size(first_img);
    
    % 初始化全聚焦图像
    focus_stacked = zeros(height, width, channels);
    
    % 读取所有图像到内存
    image_stack = cell(length(image_files), 1);
    for i = 1:length(image_files)
        img = imread(fullfile(original_path, image_files(i).name));
        image_stack{i} = double(img);
    end
    
    % 对每个像素，从最佳聚焦深度的图像中提取该像素值
    for y = 1:height
        for x = 1:width
            best_z_index = depth_map(y, x);
            % 找到最接近的图像索引
            [~, img_idx] = min(abs(indices - best_z_index));
            focus_stacked(y, x, :) = image_stack{img_idx}(y, x, :);
        end
    end
    
    focus_stacked = uint8(focus_stacked);
    
    % 绘制全聚焦图像
    fig = figure('Position', [100, 100, 1000, 800], 'Color', 'w');
    imshow(focus_stacked);
    title('Focus-Stacked Image (All-in-Focus)', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存
    saveas(fig, fullfile(output_dir, 'Fig3A_Focus_Stacked.png'));
    saveas(fig, fullfile(output_dir, 'Fig3A_Focus_Stacked.svg'));
    
    fprintf('   全聚焦图像已保存\n');
end

%% 图3B: Jet色图的3D模型
function generate_3d_jet_model(depth_map, top_result, bottom_result, colors, output_dir, height, width)
    fprintf('   3B. 生成Jet色图3D模型...\n');
    
    % 调整坐标系：top为0，bottom为负值
    depth_range = abs(top_result.peak_index - bottom_result.peak_index);
    
    % 归一化：将depth_map映射到[0, -h]区间
    min_depth = min(depth_map(:));
    max_depth = max(depth_map(:));
    
    % 反转深度：top(小index)→0，bottom(大index)→-h
    depth_normalized = -(depth_map - min_depth) / (max_depth - min_depth) * depth_range * 0.005;
    
    % 应用双峰分割 + 平面拟合降噪
    depth_filtered = bilateral_plane_filtering(depth_normalized);
    
    % 绘制3D Jet色图模型
    fig = figure('Position', [100, 100, 1200, 900], 'Color', 'w');
    
    % 3D表面图
    subplot(2, 2, [1, 2]);
    [X, Y] = meshgrid(1:width, 1:height);
    surf(X, Y, depth_filtered, 'EdgeColor', 'none', 'FaceColor', 'interp');
    colormap(jet);
    cb = colorbar;
    cb.Label.String = 'Depth (mm)';
    cb.Label.FontSize = 12;
    cb.Label.FontWeight = 'bold';
    
    title('3D Topography (Jet Colormap)', 'FontSize', 14, 'FontWeight', 'bold');
    view(45, 30);
    axis tight;
    axis off;  % 关闭坐标轴显示
    lighting gouraud;
    camlight;
    set(gca, 'FontSize', 11, 'LineWidth', 1.5);
    
    % 深度图俯视图
    subplot(2, 2, 3);
    imagesc(depth_filtered);
    colormap(gca, jet);
    colorbar;
    axis image;
    xlabel('X (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Y (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Depth Map (Top View)', 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'FontSize', 11, 'LineWidth', 1.5);
    
    % 深度分布直方图
    subplot(2, 2, 4);
    histogram(depth_filtered(:), 50, 'FaceColor', colors.primary, ...
        'EdgeColor', 'k', 'LineWidth', 1, 'FaceAlpha', 0.7);
    xlabel('Depth (mm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Pixel Count', 'FontSize', 12, 'FontWeight', 'bold');
    title('Depth Distribution', 'FontSize', 13, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'FontSize', 11, 'LineWidth', 1.5);
    
    % 输出统计信息
    fprintf('   深度统计:\n');
    fprintf('   - Top表面 (z=0): %.3f mm\n', max(depth_filtered(:)));
    fprintf('   - Bottom表面: %.3f mm\n', min(depth_filtered(:)));
    fprintf('   - 平均深度: %.3f mm\n', mean(depth_filtered(:)));
    fprintf('   - 深度范围: %.3f mm\n', range(depth_filtered(:)));
    
    % 保存
    saveas(fig, fullfile(output_dir, 'Fig3B_3D_Jet_Model.png'));
    saveas(fig, fullfile(output_dir, 'Fig3B_3D_Jet_Model.svg'));
    
    fprintf('   Jet色图3D模型已保存\n');
end

%% 图3C: RGB纹理的3D模型
function generate_3d_rgb_textured_model(original_path, depth_map, top_result, bottom_result, colors, output_dir, image_files, indices, height, width)
    fprintf('   3C. 生成RGB纹理3D模型...\n');
    
    % 读取第一张图像确定通道数
    first_img = imread(fullfile(original_path, image_files(1).name));
    channels = size(first_img, 3);
    
    % 生成全聚焦RGB图像作为纹理
    texture_image = zeros(height, width, channels);
    
    % 读取所有图像
    image_stack = cell(length(image_files), 1);
    for i = 1:length(image_files)
        img = imread(fullfile(original_path, image_files(i).name));
        image_stack{i} = double(img);
    end
    
    % 景深合成
    for y = 1:height
        for x = 1:width
            best_z_index = depth_map(y, x);
            [~, img_idx] = min(abs(indices - best_z_index));
            texture_image(y, x, :) = image_stack{img_idx}(y, x, :);
        end
    end
    
    texture_image = uint8(texture_image);
    
    % 调整深度坐标系
    depth_range = abs(top_result.peak_index - bottom_result.peak_index);
    min_depth = min(depth_map(:));
    max_depth = max(depth_map(:));
    depth_normalized = -(depth_map - min_depth) / (max_depth - min_depth) * depth_range * 0.005;
    
    % 应用双峰分割 + 平面拟合降噪
    depth_filtered = bilateral_plane_filtering(depth_normalized);
    
    % 绘制带RGB纹理的3D模型
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'w');
    
    [X, Y] = meshgrid(1:width, 1:height);
    
    % 使用surf绘制，并应用RGB纹理
    surf(X, Y, depth_filtered, texture_image, ...
        'EdgeColor', 'none', 'FaceColor', 'texturemap');
    
    title('3D Topography with RGB Texture', 'FontSize', 14, 'FontWeight', 'bold');
    view(45, 30);
    axis tight;
    axis off;  % 关闭坐标轴显示
    lighting gouraud;
    camlight('headlight');
    material dull;
    set(gca, 'FontSize', 11, 'LineWidth', 1.5);
    
    % 保存
    saveas(fig, fullfile(output_dir, 'Fig3C_3D_RGB_Textured.png'));
    saveas(fig, fullfile(output_dir, 'Fig3C_3D_RGB_Textured.svg'));
    
    fprintf('   RGB纹理3D模型已保存\n');
end

%% 辅助函数：获取排序后的图像文件
function [image_files, indices] = get_sorted_images(folder_path)
    image_extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif'};
    image_files = [];
    
    for i = 1:length(image_extensions)
        files = dir(fullfile(folder_path, image_extensions{i}));
        image_files = [image_files; files];
    end
    
    if isempty(image_files)
        error('找不到图像文件');
    end
    
    % 提取文件索引并排序
    indices = zeros(length(image_files), 1);
    for i = 1:length(image_files)
        numbers = regexp(image_files(i).name, '\d+', 'match');
        if ~isempty(numbers)
            indices(i) = str2double(numbers{1});
        end
    end
    [indices, sort_idx] = sort(indices);
    image_files = image_files(sort_idx);
end

%% 辅助函数：计算深度图
function [depth_map, focus_map] = calculate_depth_map(original_path, image_files, indices, height, width)
    fprintf('   正在计算深度图...\n');
    
    depth_map = zeros(height, width);
    focus_map = zeros(height, width);
    
    % 读取所有图像到栈
    image_stack = zeros(height, width, length(image_files));
    for i = 1:length(image_files)
        img = imread(fullfile(original_path, image_files(i).name));
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        image_stack(:, :, i) = double(img);
    end
    
    % 计算每个像素的最佳聚焦位置
    window_size = 15;
    half_win = floor(window_size / 2);
    
    total_pixels = height * width;
    processed = 0;
    last_percent = 0;
    
    for y = 1:height
        for x = 1:width
            y_start = max(1, y - half_win);
            y_end = min(height, y + half_win);
            x_start = max(1, x - half_win);
            x_end = min(width, x + half_win);
            
            focus_values = zeros(length(image_files), 1);
            for z = 1:length(image_files)
                patch = image_stack(y_start:y_end, x_start:x_end, z);
                focus_values(z) = std(patch(:));
            end
            
            [max_focus, best_z] = max(focus_values);
            depth_map(y, x) = indices(best_z);
            focus_map(y, x) = max_focus;
            
            processed = processed + 1;
            current_percent = floor(100 * processed / total_pixels);
            if current_percent > last_percent && mod(current_percent, 10) == 0
                fprintf('   进度: %d%%\n', current_percent);
                last_percent = current_percent;
            end
        end
    end
    
    fprintf('   深度图计算完成！\n');
end

%% 辅助函数：Lorentzian函数
function y = lorentzian_func(x, params)
    A = params(1);
    z0 = params(2);
    gamma = params(3);
    y = A ./ (1 + ((x - z0) / gamma).^2);
end

%% 辅助函数：Gaussian拟合
function params = fit_gaussian(x, y)
    [max_val, max_idx] = max(y);
    initial_guess = [max_val, x(max_idx), length(x)/6];
    fit_func = @(p, x) p(1) * exp(-((x - p(2)).^2) / (2 * p(3)^2));
    options = optimoptions('lsqcurvefit', 'Display', 'off');
    params = lsqcurvefit(fit_func, initial_guess, x, y, [], [], options);
end

%% 辅助函数：计算R²
function r2 = calculate_r_squared(y_true, y_pred)
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2 = 1 - (ss_res / ss_tot);
end

%% 辅助函数：分析文件夹聚焦度
function result = analyze_folder_focus(folder_path)
    result = [];
    
    if ~exist(folder_path, 'dir')
        return;
    end
    
    image_extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif'};
    image_files = [];
    
    for i = 1:length(image_extensions)
        files = dir(fullfile(folder_path, image_extensions{i}));
        image_files = [image_files; files];
    end
    
    if length(image_files) < 3
        return;
    end
    
    indices = zeros(length(image_files), 1);
    focus_measures = zeros(length(image_files), 1);
    valid_count = 0;
    
    for i = 1:length(image_files)
        filename = image_files(i).name;
        numbers = regexp(filename, '\d+', 'match');
        if isempty(numbers)
            continue;
        end
        index = str2double(numbers{1});
        
        image_path = fullfile(folder_path, filename);
        try
            img = imread(image_path);
            if isempty(img)
                continue;
            end
            
            focus_measure = wavelet_focus_measure(img);
            
            valid_count = valid_count + 1;
            indices(valid_count) = index;
            focus_measures(valid_count) = focus_measure;
        catch
            continue;
        end
    end
    
    indices = indices(1:valid_count);
    focus_measures = focus_measures(1:valid_count);
    
    if valid_count < 3
        return;
    end
    
    [indices, sort_idx] = sort(indices);
    focus_measures = focus_measures(sort_idx);
    
    try
        [max_focus, max_idx] = max(focus_measures);
        max_index = indices(max_idx);
        
        half_max = max_focus / 2;
        above_half = focus_measures > half_max;
        if sum(above_half) >= 2
            half_indices = indices(above_half);
            gamma_estimate = (max(half_indices) - min(half_indices)) / 2;
        else
            gamma_estimate = length(indices) / 6;
        end
        
        initial_guess = [max_focus, max_index, gamma_estimate];
        lb = [0, min(indices), 0.1];
        ub = [inf, max(indices), range(indices)];
        
        fit_func = @(params, x) lorentzian_func(x, params);
        options = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxIterations', 1000);
        [popt, ~, residual, ~, ~, ~, jacobian] = lsqcurvefit(fit_func, initial_guess, indices, focus_measures, lb, ub, options);
        
        peak_index = popt(2);
        x_smooth = linspace(min(indices), max(indices), 1000);
        y_smooth = lorentzian_func(x_smooth, popt);
        
        y_pred = lorentzian_func(indices, popt);
        ss_res = sum((focus_measures - y_pred).^2);
        ss_tot = sum((focus_measures - mean(focus_measures)).^2);
        r_squared = 1 - (ss_res / ss_tot);
        rmse = sqrt(mean((focus_measures - y_pred).^2));
        
        ci = nlparci(popt, residual, 'jacobian', jacobian);
        param_errors = (ci(:,2) - ci(:,1)) / (2 * 1.96);
        
        fwhm = 2 * popt(3);
        
        result.peak_index = peak_index;
        result.r_squared = r_squared;
        result.rmse = rmse;
        result.params = popt;
        result.param_errors = param_errors;
        result.data_points = valid_count;
        result.indices = indices;
        result.focus_measures = focus_measures;
        result.x_smooth = x_smooth;
        result.y_smooth = y_smooth;
        result.fwhm = fwhm;
    catch
        return;
    end
end

%% 辅助函数：小波聚焦度测量
function focus_measure = wavelet_focus_measure(image)
    if size(image, 3) == 3
        gray = rgb2gray(image);
    else
        gray = image;
    end
    
    gray = double(gray);
    [C, S] = wavedec2(gray, 3, 'db2');
    
    focus_measure = 0.0;
    for level = 1:3
        [cH, cV, cD] = detcoef2('all', C, S, level);
        focus_measure = focus_measure + sum(cH(:).^2) + sum(cV(:).^2) + sum(cD(:).^2);
    end
    
    focus_measure = focus_measure / numel(gray);
end

%% 辅助函数：双峰分割 + 平面拟合降噪
function depth_filtered = bilateral_plane_filtering(depth_map)
    % 基于物理结构的降噪：识别top和bottom两个平面，分别拉平
    
    % 步骤1: 计算深度直方图，识别双峰
    [counts, edges] = histcounts(depth_map(:), 100);
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    
    % 平滑直方图以找峰
    counts_smooth = smooth(counts, 5);
    
    % 找到两个主峰（top和bottom）
    [pks, locs] = findpeaks(counts_smooth, 'MinPeakDistance', 20, 'SortStr', 'descend');
    
    if length(locs) >= 2
        % 有明显双峰：分别处理top和bottom平面
        peak_depths = centers(locs(1:2));
        peak_depths = sort(peak_depths, 'descend'); % peak_depths(1)是top(接近0), peak_depths(2)是bottom(接近-h)
        
        % 找到两个峰之间的谷点作为分界
        [~, valley_idx] = min(counts_smooth(min(locs(1:2)):max(locs(1:2))));
        threshold = centers(min(locs(1:2)) + valley_idx - 1);
        
        % 分离top和bottom区域
        top_mask = depth_map > threshold;
        bottom_mask = depth_map <= threshold;
        
        % 初始化输出
        depth_filtered = depth_map;
        
        % Top平面区域：拉平到均值
        if sum(top_mask(:)) > 0
            top_mean = mean(depth_map(top_mask));
            depth_filtered(top_mask) = top_mean;
        end
        
        % Bottom平面区域：拉平到均值
        if sum(bottom_mask(:)) > 0
            bottom_mean = mean(depth_map(bottom_mask));
            depth_filtered(bottom_mask) = bottom_mean;
        end
        
        % 过渡区域（微孔壁）：识别为梯度较大的区域，轻度平滑
        [Gx, Gy] = gradient(depth_map);
        gradient_mag = sqrt(Gx.^2 + Gy.^2);
        transition_mask = gradient_mag > median(gradient_mag(:)) * 2;
        
        % 对过渡区域用小窗口中值滤波
        if sum(transition_mask(:)) > 0
            depth_temp = medfilt2(depth_map, [3, 3]);
            depth_filtered(transition_mask) = depth_temp(transition_mask);
        end
        
    else
        % 没有明显双峰：退回到轻度平滑
        fprintf('   警告: 未检测到明显的双峰分布，使用标准降噪\n');
        depth_filtered = medfilt2(depth_map, [3, 3]);
    end
    
    % 最后整体做一次轻微的双边滤波，消除边界处的跳变
    depth_filtered = imbilatfilt(depth_filtered, 0.01, 2);
end