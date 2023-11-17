export default function checkFourAdjacent(matrix: any[][]): boolean {
    const rows = matrix.length;
    const cols = matrix[0].length;

    function checkAdjacent(a: any, b: any, c: any, d: any): boolean {
        return (a == 1 && b == 1 && c == 1 && d == 1) || (a == -1 && b == -1 && c == -1 && d == -1)
    }

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols - 3; j++) {
            if (checkAdjacent(matrix[i][j], matrix[i][j + 1], matrix[i][j + 2], matrix[i][j + 3])) {
                return true;
            }
        }
    }

    for (let i = 0; i < rows - 3; i++) {
        for (let j = 0; j < cols; j++) {
            if (checkAdjacent(matrix[i][j], matrix[i + 1][j], matrix[i + 2][j], matrix[i + 3][j])) {
                return true;
            }
        }
    }

    for (let i = 3; i < rows; i++) {
        for (let j = 0; j < cols - 3; j++) {
            if (checkAdjacent(matrix[i][j], matrix[i - 1][j + 1], matrix[i - 2][j + 2], matrix[i - 3][j + 3])) {
                return true;
            }
        }
    }

    for (let i = 0; i < rows - 3; i++) {
        for (let j = 0; j < cols - 3; j++) {
            if (checkAdjacent(matrix[i][j], matrix[i + 1][j + 1], matrix[i + 2][j + 2], matrix[i + 3][j + 3])) {
                return true;
            }
        }
    }

    return false;
}
