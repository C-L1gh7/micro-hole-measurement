function focus_analysis_main()
    % 聚焦度分析主程序
    % 用于分析图像的聚焦度并计算深度信息
    
    % 选择文件夹（GUI方式）
    base_path = uigetdir('', '请选择包含processed和original文件夹的目录');
    if base_path == 0
        disp('用户取消操作');
        return;
    end
    
    % 检查文件夹结构
    processed_path = fullfile(base_path, 'processed');
    original_path = fullfile(base_path, 'original');
    
    if ~exist(processed_path, 'dir')
        error('找不到processed文件夹：%s', processed_path);
    end
    
    if ~exist(original_path, 'dir')
        error('找不到original文件夹：%s', original_path);
    end
    
    % 分析top和bottom文件夹
    top_folder = fullfile(processed_path, 'top');
    bottom_folder = fullfile(processed_path, 'bottom');
    
    fprintf('正在分析TOP文件夹...\n');
    top_result = analyze_folder_focus(top_folder);
    
    fprintf('正在分析BOTTOM文件夹...\n');
    bottom_result = analyze_folder_focus(bottom_folder);
    
    % 提取峰值
    if ~isempty(top_result)
        top_peak = top_result.peak_index;
        fprintf('TOP最佳聚焦位置: %.3f\n', top_peak);
        fprintf('TOP拟合R²: %.3f\n', top_result.r_squared);
        fprintf('TOP拟合RMSE: %.3f\n', top_result.rmse);
        fprintf('TOP半高全宽(FWHM): %.3f\n', top_result.fwhm);
    else
        top_peak = [];
        fprintf('TOP最佳: 无法计算\n');
    end
    
    if ~isempty(bottom_result)
        bottom_peak = bottom_result.peak_index;
        fprintf('BOTTOM最佳聚焦位置: %.3f\n', bottom_peak);
        fprintf('BOTTOM拟合R²: %.3f\n', bottom_result.r_squared);
        fprintf('BOTTOM拟合RMSE: %.3f\n', bottom_result.rmse);
        fprintf('BOTTOM半高全宽(FWHM): %.3f\n', bottom_result.fwhm);
    else
        bottom_peak = [];
        fprintf('BOTTOM最佳: 无法计算\n');
    end
    
    % 计算深度
    if ~isempty(top_peak) && ~isempty(bottom_peak)
        % 初步计算深度（未校正）
        depth_measured = (bottom_peak - top_peak) * 0.005;
        
        % 应用校正公式: d = 测得d - 测得d*(-0.09024) + 0.19598
        depth = depth_measured - (depth_measured * (-0.09024) + 0.19598);
        
        %fprintf('测得深度(未校正): %.3f mm\n', depth_measured);
        fprintf('校正后深度: %.3f mm\n', depth);
    else
        depth = [];
        fprintf('深度: 无法计算\n');
    end
    
    % 显示原始图片
    if ~isempty(top_peak)
        fprintf('\n显示TOP最佳聚焦位置附近的原始图片...\n');
        display_original_images(original_path, top_peak, 'TOP');
    end
    
    if ~isempty(bottom_peak)
        fprintf('\n显示BOTTOM最佳聚焦位置附近的原始图片...\n');
        display_original_images(original_path, bottom_peak, 'BOTTOM');
    end
    
    % 绘制拟合曲线
    figure('Name', '聚焦度分析结果', 'NumberTitle', 'off');
    
    if ~isempty(top_result)
        subplot(2, 1, 1);
        plot_focus_curve(top_result, 'TOP');
    end
    
    if ~isempty(bottom_result)
        subplot(2, 1, 2);
        plot_focus_curve(bottom_result, 'BOTTOM');
    end
end

function focus_measure = wavelet_focus_measure(image)
    % 使用小波变换计算聚焦度
    % 输入: 图像矩阵
    % 输出: 聚焦度值
    
    % 转换为灰度图
    if size(image, 3) == 3
        gray = rgb2gray(image);
    else
        gray = image;
    end
    
    % 转换为double类型
    gray = double(gray);
    
    % 3层小波分解
    [C, S] = wavedec2(gray, 3, 'db2');
    
    % 计算高频系数的能量
    focus_measure = 0.0;
    
    % 从第1层到第3层提取高频系数
    for level = 1:3
        % 提取对角、水平、垂直细节系数
        [cH, cV, cD] = detcoef2('all', C, S, level);
        
        % 计算能量
        focus_measure = focus_measure + sum(cH(:).^2) + sum(cV(:).^2) + sum(cD(:).^2);
    end
    
    % 归一化
    focus_measure = focus_measure / numel(gray);
end

function y = lorentzian(x, params)
    % Lorentzian (Cauchy) 函数
    % 更适合显微镜的Airy PSF模型
    % params = [A, z0, gamma]
    % M(z) = A / (1 + ((z-z0)/gamma)^2)
    
    A = params(1);      % 峰值振幅
    z0 = params(2);     % 峰值位置
    gamma = params(3);  % 半高全宽(FWHM)的一半
    
    y = A ./ (1 + ((x - z0) / gamma).^2);
end

function result = analyze_folder_focus(folder_path)
    % 分析文件夹中图像的聚焦度并进行Lorentzian拟合
    
    result = [];
    
    if ~exist(folder_path, 'dir')
        fprintf('文件夹不存在: %s\n', folder_path);
        return;
    end
    
    % 获取所有图像文件
    image_extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif'};
    image_files = [];
    
    for i = 1:length(image_extensions)
        files = dir(fullfile(folder_path, image_extensions{i}));
        image_files = [image_files; files];
    end
    
    if length(image_files) < 3
        fprintf('图像文件数量不足（至少需要3张）\n');
        return;
    end
    
    % 提取文件名中的数字并排序
    indices = zeros(length(image_files), 1);
    focus_measures = zeros(length(image_files), 1);
    valid_count = 0;
    
    for i = 1:length(image_files)
        filename = image_files(i).name;
        
        % 提取数字
        numbers = regexp(filename, '\d+', 'match');
        if isempty(numbers)
            continue;
        end
        index = str2double(numbers{1});
        
        % 读取图像
        image_path = fullfile(folder_path, filename);
        try
            img = imread(image_path);
            if isempty(img)
                continue;
            end
            
            % 计算聚焦度
            focus_measure = wavelet_focus_measure(img);
            
            valid_count = valid_count + 1;
            indices(valid_count) = index;
            focus_measures(valid_count) = focus_measure;
        catch
            continue;
        end
    end
    
    % 截取有效数据
    indices = indices(1:valid_count);
    focus_measures = focus_measures(1:valid_count);
    
    if valid_count < 3
        fprintf('有效图像文件数量不足\n');
        return;
    end
    
    % 按索引排序
    [indices, sort_idx] = sort(indices);
    focus_measures = focus_measures(sort_idx);
    
    % Lorentzian拟合
    try
        % 初始参数估计
        [max_focus, max_idx] = max(focus_measures);
        max_index = indices(max_idx);
        
        % 估计半高全宽（FWHM的一半）
        % 找到半高点估计gamma
        half_max = max_focus / 2;
        above_half = focus_measures > half_max;
        if sum(above_half) >= 2
            half_indices = indices(above_half);
            gamma_estimate = (max(half_indices) - min(half_indices)) / 2;
        else
            gamma_estimate = length(indices) / 6;
        end
        
        % 初始参数: [A, z0, gamma]
        initial_guess = [max_focus, max_index, gamma_estimate];
        
        % 设置参数边界（避免负值和不合理的值）
        lb = [0, min(indices), 0.1];  % 下界
        ub = [inf, max(indices), range(indices)];  % 上界
        
        % 定义拟合函数
        fit_func = @(params, x) lorentzian(x, params);
        
        % 使用lsqcurvefit进行拟合
        options = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxIterations', 1000);
        [popt, ~, residual, ~, ~, ~, jacobian] = lsqcurvefit(fit_func, initial_guess, indices, focus_measures, lb, ub, options);
        
        % Lorentzian函数的峰值位置就是z0
        peak_index = popt(2);
        
        % 生成平滑曲线用于绘图
        x_smooth = linspace(min(indices), max(indices), 1000);
        y_smooth = lorentzian(x_smooth, popt);
        
        % 计算拟合统计量
        y_pred = lorentzian(indices, popt);
        ss_res = sum((focus_measures - y_pred).^2);
        ss_tot = sum((focus_measures - mean(focus_measures)).^2);
        r_squared = 1 - (ss_res / ss_tot);
        rmse = sqrt(mean((focus_measures - y_pred).^2));
        
        % 计算参数标准误差
        ci = nlparci(popt, residual, 'jacobian', jacobian);
        param_errors = (ci(:,2) - ci(:,1)) / (2 * 1.96);  % 标准误差
        
        % 计算FWHM（半高全宽）
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
        result.fwhm = fwhm;  % 添加半高全宽信息
        
    catch ME
        fprintf('拟合失败: %s\n', ME.message);
        return;
    end
end

function plot_focus_curve(result, title_str)
    % 绘制聚焦度曲线和Lorentzian拟合结果
    
    hold on;
    plot(result.indices, result.focus_measures, 'bo', 'MarkerSize', 8, 'DisplayName', '实测数据');
    plot(result.x_smooth, result.y_smooth, 'r-', 'LineWidth', 2, 'DisplayName', 'Lorentzian拟合');
    plot(result.peak_index, max(result.y_smooth), 'r*', 'MarkerSize', 15, 'DisplayName', '峰值位置');
    
    % 标注FWHM
    peak_height = max(result.y_smooth);
    half_height = peak_height / 2;
    plot([min(result.x_smooth), max(result.x_smooth)], [half_height, half_height], ...
        'g--', 'LineWidth', 1, 'DisplayName', '半高线');
    
    xlabel('图像索引 (深度位置)');
    ylabel('聚焦度');
    title(sprintf('%s - Lorentzian拟合\n峰值: %.3f, R²: %.3f, FWHM: %.3f', ...
        title_str, result.peak_index, result.r_squared, result.fwhm));
    legend('Location', 'best');
    grid on;
    hold off;
end

function display_original_images(original_path, peak_index, label)
    % 显示最佳聚焦位置附近的原始图片
    
    center_frame = round(peak_index);
    offsets = -2:2;  % 显示前后各2帧
    
    figure('Name', sprintf('%s - 原始图片', label), 'NumberTitle', 'off');
    
    for i = 1:length(offsets)
        idx = center_frame + offsets(i);
        image_path = fullfile(original_path, sprintf('%d.png', idx));
        
        if ~exist(image_path, 'file')
            % 尝试其他扩展名
            image_path = fullfile(original_path, sprintf('%d.jpg', idx));
        end
        
        if exist(image_path, 'file')
            img = imread(image_path);
            subplot(1, 5, i);
            imshow(img);
            if offsets(i) == 0
                title(sprintf('%d (最佳)', idx), 'FontWeight', 'bold', 'Color', 'r');
            else
                title(sprintf('%d', idx));
            end
        end
    end
end