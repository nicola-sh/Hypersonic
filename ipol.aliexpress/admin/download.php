<?php
use Bitrix\Main\Loader;

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_admin.php");

Loader::includeModule('ipol.aliexpress');

$orderIds  = is_array($_REQUEST['IDS'])   ? $_REQUEST['IDS'] : explode(',', $_REQUEST['IDS']);
$fileTypes = is_array($_REQUEST['TYPES']) ? $_REQUEST['TYPES'] : explode(',', $_REQUEST['TYPES'] ?: 'all');
$files     = [];

foreach ($orderIds as $orderId) {
    $order = \Ipol\Aliexpress\DB\OrderTable::getByPrimary($orderId, ['select' => ['*']])->fetchObject();

    if (!$order || !$order->isCreated()) {
        continue;
    }

    foreach (['doc', 'label'] as $fileType) {
        if (!in_array($fileType, $fileTypes) && !in_array('all', $fileTypes)) {
            continue;
        }

        if ($fileType == 'doc') {
            $getFileResult = $order->aliexpress()->getPrintDoc();
        } else {
            $getFileResult = $order->aliexpress()->getPrintLabel();
        }

        if (!$getFileResult->isSuccess()) {
            continue;
        }

        $data                = $getFileResult->getData();
        $entryName           = $order['ALI_LP_NUMBER'] .'_'. $fileType .'.pdf';
        $files[ $entryName ] = $data['file'];
    }
}

if (empty($files)) {
    \ShowError('Files not found');
    exit;
}

$zipFile = tempnam('/tmp', 'ipol_ali_');
$zip     = new ZipArchive();

if (!$zip->open($zipFile, ZipArchive::CREATE) !== false) {
    \ShowError('Unable create zip archive');
    exit;
}

foreach ($files as $entryName => $fileName) {
    $zip->addFile($_SERVER['DOCUMENT_ROOT'] . $fileName, $entryName);
}

$zip->close();

$GLOBALS['APPLICATION']->RestartBuffer();

header("Content-type:application/zip");
header("Content-Disposition:attachment;filename=download.zip");

readfile($zipFile);