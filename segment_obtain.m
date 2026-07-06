function Segment_matrix = segment_obtain(EEG, length)
    [n, m] = size(EEG);
    len = m - length + 1;  % 每行能生成的片段数

    % 预分配
    Segment_matrix = zeros(n * len, length);

    % 利用矩阵索引批量提取滑动窗口片段
    idx = 1;
    for j = 1:n
        % 使用 im2col 来生成每行的滑动窗口
        segments = im2col(EEG(j, :), [1, length], 'sliding')';
        Segment_matrix(idx:idx + len - 1, :) = segments;
        idx = idx + len;
    end
end
